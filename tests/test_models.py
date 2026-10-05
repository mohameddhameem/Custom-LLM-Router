"""Exercise the transformers code paths with tiny randomly initialised local models."""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from tokenizers import Tokenizer, models, pre_tokenizers, processors  # noqa: E402
from transformers import (  # noqa: E402
    BertConfig,
    BertForSequenceClassification,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)

from groupproject import evaluate_routing, make_splits, router, run_experts  # noqa: E402
from groupproject.evidence import CrossEncoderScorer, RerankerScorer  # noqa: E402
from groupproject.experts import PROMPT, HFExpert  # noqa: E402
from synthetic import make_split  # noqa: E402

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[EOS]"]


def build_tokenizer(texts: list[str], pair_template: bool) -> PreTrainedTokenizerFast:
    pre = pre_tokenizers.Whitespace()
    words = sorted({w for t in texts for w, _ in pre.pre_tokenize_str(t)})
    vocab = {tok: i for i, tok in enumerate(SPECIALS + words)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre
    if pair_template:
        tok.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
        )
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", eos_token="[EOS]")


@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    df = make_split(50, seed=3)
    texts = [PROMPT, "question: option: irrelevant partially relevant directly answers yes no user assistant",
             CrossEncoderScorer.RELEVANCE[0], CrossEncoderScorer.SUFFICIENT[0]]
    texts += df["question"].tolist() + df["full_context"].tolist()

    lm_tok = build_tokenizer(texts, pair_template=False)
    lm = GPT2LMHeadModel(GPT2Config(vocab_size=len(lm_tok), n_positions=1024, n_embd=32, n_layer=1, n_head=2,
                                    bos_token_id=lm_tok.eos_token_id, eos_token_id=lm_tok.eos_token_id))
    lm.save_pretrained(root / "lm")
    lm_tok.save_pretrained(root / "lm")

    chat_tok = build_tokenizer(texts, pair_template=False)
    chat_tok.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}assistant:"
    lm.save_pretrained(root / "chat-lm")
    chat_tok.save_pretrained(root / "chat-lm")

    ce_tok = build_tokenizer(texts, pair_template=True)
    ce = BertForSequenceClassification(BertConfig(vocab_size=len(ce_tok), hidden_size=32, num_hidden_layers=1,
                                                  num_attention_heads=2, intermediate_size=64, num_labels=1))
    ce.save_pretrained(root / "ce")
    ce_tok.save_pretrained(root / "ce")
    (root / "ce" / "calibration.json").write_text(json.dumps({"relevance": 1.3, "sufficient": 1.35}))
    return root


@pytest.mark.parametrize("model_dir", ["lm", "chat-lm"])
def test_hf_expert_answers(tiny_models, model_dir):
    expert = HFExpert("small", 2, str(tiny_models / model_dir), device="cpu", max_new_tokens=4, dtype="float32")
    out = expert.answer("Who directed the film?", ["Film: It was directed by Kalo.", "Kalo: Kalo was born in Rimer."])
    assert isinstance(out.answer, str)
    assert out.cost > 10
    assert np.isfinite(out.uncertainty) and out.uncertainty >= 0


def test_cross_encoder_scores(tiny_models):
    scorer = CrossEncoderScorer(str(tiny_models / "ce"), device="cpu")
    assert scorer.temperatures == {"relevance": 1.3, "sufficient": 1.35}
    scores = scorer.score("Who directed the film?", ["Film: It was directed by Kalo.", "Other: Unrelated.", "X: y."])
    assert scores.shape == (3,) and ((scores > 0) & (scores < 1)).all()
    assert 0 < scorer.sufficiency("Who directed the film?", ["Film: It was directed by Kalo."]) < 1
    seen = []
    scorer._decide = lambda decision, q, states: seen.append(states) or np.array([[0.5, 0.5]])
    scorer.sufficiency("q", ["A: one.", "B: two."])
    assert seen == [["[1] A: one.\n[2] B: two."]]  # nano-jev's format_passages layout
    assert scorer.pairs_seen == 3 * 3 + 2


def test_reranker_scores(tiny_models):
    scorer = RerankerScorer(str(tiny_models / "ce"), device="cpu")
    scores = scorer.score("Who directed the film?", ["Film: It was directed by Kalo.", "Other: Unrelated."])
    assert scores.shape == (2,) and ((scores > 0) & (scores < 1)).all()
    assert np.isnan(scorer.sufficiency("Who directed the film?", ["Film: x"]))


def test_end_to_end_with_models(tiny_models, tmp_path):
    config = tmp_path / "models.toml"
    config.write_text(f"""
[scorer]
kind = "cross-encoder"
path = "{tiny_models / 'ce'}"
device = "cpu"
sufficiency_top_k = 2

[[experts]]
name = "small"
kind = "hf"
model = "{tiny_models / 'lm'}"
top_k = 2
device = "cpu"
max_new_tokens = 4
dtype = "float32"

[[experts]]
name = "large"
kind = "hf"
model = "{tiny_models / 'chat-lm'}"
device = "cpu"
max_new_tokens = 4
dtype = "float32"
""")
    train, val = make_split(40, seed=5, level="medium"), make_split(15, seed=6, start=500)
    train.to_parquet(tmp_path / "train.parquet")
    val.to_parquet(tmp_path / "val.parquet")
    make_splits.main(["--train", str(tmp_path / "train.parquet"), "--router-size", "25", "--calib-size", "10",
                      "--out", str(tmp_path / "splits.json")])
    run = tmp_path / "run"
    common = ["--config", str(config), "--run", str(run)]
    for split in ("router_train", "calib"):
        run_experts.main(common + ["--data", str(tmp_path / "train.parquet"), "--splits", str(tmp_path / "splits.json"),
                                   "--split", split, "--name", split])
    run_experts.main(common + ["--data", str(tmp_path / "val.parquet"), "--name", "test"])

    cache = pd.read_parquet(run / "router_train.parquet")
    assert cache["ev_sufficiency"].between(0, 1).all()
    assert (cache["large_cost"] > cache["small_cost"]).all()
    # random models never answer correctly; plant correct small answers so both labels occur
    cache.loc[cache.index[::2], "small_f1"] = 1.0
    cache.to_parquet(run / "router_train.parquet")

    router.main(["--run", str(run)])
    evaluate_routing.main(["--run", str(run), "--cost", "tokens"])  # local model paths carry no size
    report = json.loads((run / "report.json").read_text())
    assert report["n_test"] == 15
    assert "ev_sufficiency" in router_columns(run)


def router_columns(run) -> list[str]:
    import pickle

    return pickle.loads((run / "routers.pkl").read_bytes())["routers"]["question+evidence/small-fails"]["columns"]


@pytest.mark.parametrize("model_dir", ["lm", "chat-lm"])
def test_hf_expert_batches_match_single(tiny_models, model_dir):
    qs = ["Who directed the film?", "Where was Kalo born?", "Which film came out first, Film or Other?"]
    ctx = [["Film: It was directed by Kalo."], ["Kalo: Kalo was born in Rimer.", "Other: Unrelated text here."],
           ["Film: A 1990 film."]]
    single = HFExpert("s", None, str(tiny_models / model_dir), device="cpu", max_new_tokens=5, dtype="float32")
    batched = HFExpert("s", None, str(tiny_models / model_dir), device="cpu", max_new_tokens=5, dtype="float32",
                       batch_size=3)
    one = [single.answer(q, c) for q, c in zip(qs, ctx)]
    many = batched.answer_batch(qs, ctx)
    assert [o.answer for o in one] == [o.answer for o in many]
    assert [o.cost for o in one] == [o.cost for o in many]
    assert [o.uncertainty for o in one] == pytest.approx([o.uncertainty for o in many], rel=1e-3)
