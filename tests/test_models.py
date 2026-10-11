"""Exercise the transformers code paths with tiny randomly initialised local models."""

import json
from pathlib import Path

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

from stair import evaluate_routing, make_splits, router, run_experts  # noqa: E402
from stair.evidence import CrossEncoderScorer, RerankerScorer  # noqa: E402
from stair.experts import PROMPT, HFExpert  # noqa: E402
from synthetic import make_split  # noqa: E402

CONFIGS = Path(__file__).parent.parent / "configs"
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

    # same weights, but a generation config that would change greedy output (as Qwen2.5's does)
    lm.generation_config.repetition_penalty = 100.0
    lm.save_pretrained(root / "lm-penalty")
    lm_tok.save_pretrained(root / "lm-penalty")

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
    assert out.prompt_tokens + out.generated_tokens == out.cost
    assert len(out.token_entropy) == out.generated_tokens
    assert out.uncertainty == pytest.approx(np.mean(out.token_entropy))


def test_hf_expert_ignores_model_generation_config(tiny_models):
    """Plain greedy decoding: a repetition penalty in the model's config must not change answers or entropy."""
    qs = ["Who directed the film?", "Where was Kalo born?"]
    ctx = [["Film: It was directed by Kalo. Kalo Kalo Kalo."], ["Kalo: Kalo was born in Rimer."]]
    plain = HFExpert("s", None, str(tiny_models / "lm"), device="cpu", max_new_tokens=8, dtype="float32")
    penalised = HFExpert("s", None, str(tiny_models / "lm-penalty"), device="cpu", max_new_tokens=8, dtype="float32")
    a, b = plain.answer_batch(qs, ctx), penalised.answer_batch(qs, ctx)
    assert [o.answer for o in a] == [o.answer for o in b]
    assert [o.token_entropy for o in a] == [pytest.approx(o.token_entropy) for o in b]


def test_hf_expert_rejects_overlong_prompts(tiny_models):
    expert = HFExpert("s", None, str(tiny_models / "lm"), device="cpu", max_new_tokens=4, dtype="float32")
    with pytest.raises(ValueError, match="exceed"):
        expert.answer("Who directed the film?", ["Film: It was directed by Kalo. " * 200])


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
path = "{(tiny_models / 'ce').as_posix()}"
device = "cpu"
sufficiency_top_k = 2

[[experts]]
name = "small"
kind = "hf"
model = "{(tiny_models / 'lm').as_posix()}"
top_k = 2
device = "cpu"
max_new_tokens = 4
dtype = "float32"

[[experts]]
name = "large"
kind = "hf"
model = "{(tiny_models / 'chat-lm').as_posix()}"
device = "cpu"
max_new_tokens = 4
dtype = "float32"
""")
    train, val = make_split(40, seed=5, level="medium"), make_split(15, seed=6, start=500)
    train.to_parquet(tmp_path / "train.parquet")
    val.to_parquet(tmp_path / "val.parquet")
    make_splits.main(["--train", str(tmp_path / "train.parquet"), "--level", "medium", "--router-size", "25",
                      "--calib-size", "10", "--out", str(tmp_path / "splits.json")])
    run = tmp_path / "run"
    common = ["--config", str(config), "--run", str(run)]
    for split in ("router_train", "calib"):
        run_experts.main(common + ["--data", str(tmp_path / "train.parquet"), "--splits", str(tmp_path / "splits.json"),
                                   "--split", split, "--name", split])
    run_experts.main(common + ["--data", str(tmp_path / "val.parquet"), "--name", "test"])

    cache = pd.read_parquet(run / "router_train.parquet")
    assert cache["ev_sufficiency"].between(0, 1).all()
    assert (cache["ev_sufficiency"] == cache["ev_sufficiency_k2"]).all()
    assert cache["ev_sufficiency_k3"].between(0, 1).all()
    assert cache["para_scores"].map(len).eq(10).all()
    assert (cache["scorer_tokens"] > 0).all() and (cache["scorer_truncated"] >= 0).all()
    assert (cache["small_prompt_tokens"] + cache["small_generated_tokens"] == cache["small_cost"]).all()
    assert (cache["large_token_entropy"].map(len) == cache["large_generated_tokens"]).all()
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


def test_system_one_judges_end_to_end(tiny_models, tmp_path):
    from stair import judge

    data = tmp_path / "data"
    data.mkdir()
    make_split(160, seed=7, level="hard").to_parquet(data / "distractor_train.parquet")
    make_split(60, seed=8, start=5_000).to_parquet(data / "distractor_validation.parquet")
    make_splits.main(["--train", str(data / "distractor_train.parquet"), "--router-size", "100", "--calib-size", "60",
                      "--out", str(data / "splits.json")])
    run = tmp_path / "run"
    common = ["--config", str(CONFIGS / "smoke.toml"), "--run", str(run)]
    for split in ("router_train", "calib"):
        run_experts.main(common + ["--data", str(data / "distractor_train.parquet"), "--splits",
                                   str(data / "splits.json"), "--split", split, "--name", split])
    run_experts.main(common + ["--data", str(data / "distractor_validation.parquet"), "--name", "test"])

    ce = tiny_models / "ce"
    plan = tmp_path / "plan.toml"
    plan.write_text(f"""
tau = 0.8
seeds = [0, 1]
zero_shot_judge = "{ce}"
[pairs]
A = "{run}"
B = "{run}"
[train]
epochs = 2
lr = 1e-3
weight_decay = 0.0
warmup = 0.0
batch_size = 8
max_length = 128
dev_fraction = 0.2
[[judges]]
inputs = "qpa"
label = "small-fails"
init = "{ce}"
train_pair = "A"
[[judges]]
inputs = "q"
label = "small-fails"
init = "{ce}"
train_pair = "B"
""")
    out = tmp_path / "out"
    exclude = tmp_path / "exclude.txt"
    exclude.write_text("\n".join(pd.read_parquet(run / "test.parquet")["id"].head(10)))
    judge.main(["all", "--plan", str(plan), "--data-dir", str(data), "--out", str(out), "--device", "cpu",
                "--bootstrap", "30", "--exclude", str(exclude)])

    zs = pd.read_parquet(out / "zeroshot" / "A" / "test.parquet")
    assert len(zs) == 60 and zs["zs_grounded"].between(0, 1).all() and (zs["zs_correct_tokens"] > 0).all()
    done = json.loads((out / "judges" / "qpa-small-fails-ce-A" / "seed1" / "done.json").read_text())
    assert len(done["history"]) == 2 and done["n_dev"] == 20 and done["seed"] == 1
    report = json.loads((out / "eval" / "A" / "report.json").read_text())
    judges = report["judges"]
    assert set(judges) == {"zs_grounded", "zs_correct", "judge/qpa-small-fails-ce-A", "judge/q-small-fails-ce-B"}
    trained = judges["judge/qpa-small-fails-ce-A"]
    assert trained["post"] is True and len(trained["aiq_per_seed"]) == 2 and trained["judge_gflops"] > 0
    assert trained["aiq_with_judge_cost"] <= trained["aiq"] + 1e-12  # paying for the judge never helps
    assert "router:judge/qpa-small-fails-ce-A - router:question+evidence+small/large-helps" in report["bootstrap"]["aiq_diff"]
    scores = pd.read_parquet(out / "eval" / "A" / "scores-test.parquet")
    assert {"judge/qpa-small-fails-ce-A", "judge/qpa-small-fails-ce-A#seed0", "zs_grounded"} <= set(scores.columns)
    assert json.loads((out / "eval" / "B" / "report-clean.json").read_text())["n_test"] == 50

    assert done["scoring_seconds_per_question"]["B/test"] == "reused"  # pair B has the same inputs as A
    pd.testing.assert_frame_equal(pd.read_parquet(out / "zeroshot" / "A" / "test.parquet"),
                                  pd.read_parquet(out / "zeroshot" / "B" / "test.parquet"))
    before = (out / "judges" / "qpa-small-fails-ce-A" / "seed0" / "done.json").stat().st_mtime
    judge.main(["train", "--plan", str(plan), "--data-dir", str(data), "--out", str(out), "--device", "cpu"])
    assert (out / "judges" / "qpa-small-fails-ce-A" / "seed0" / "done.json").stat().st_mtime == before  # resumes


def test_micro_batches_give_the_full_batch_gradient(tiny_models):
    import torch

    from stair.judge import Judge, accumulate

    items = [{"question": f"Which model should answer this question: q{i}", "options": ("small model", "large model"),
              "state": "Film: It was directed by Kalo." * (i + 1)} for i in range(5)]
    target = torch.tensor([0, 1, 1, 0, 1])
    grads = []
    for micro in (5, 2, 1):
        judge = Judge(str(tiny_models / "ce"), "cpu", 64)
        judge.model.eval()  # no dropout, so the passes are comparable
        judge.model.zero_grad()
        loss = accumulate(judge, items, target, micro)
        grads.append((loss, torch.cat([p.grad.flatten() for p in judge.model.parameters() if p.grad is not None])))
    for loss, grad in grads[1:]:
        assert loss == pytest.approx(grads[0][0], rel=1e-5)
        assert torch.allclose(grad, grads[0][1], atol=1e-6)


def test_pools_score_and_recall(tiny_models, tmp_path):
    from stair import pools

    rows = [{"id": f"q{i}", "question": "Who directed the film?", "answer": "Kalo", "type": "bridge", "level": "hard",
             "titles": ["Film", "Other", "Kalo"], "texts": ["It was directed by Kalo.", "Unrelated.", "Kalo was born."],
             "gold_idx": [0, 2], "n_gold": 2, "bm25": [2.0, 1.0, 0.5]} for i in range(5)]
    (tmp_path / "pools").mkdir()
    pd.DataFrame(rows).to_parquet(tmp_path / "pools" / "t.parquet")
    args = ["--name", "t", "--out", str(tmp_path)]
    pools.main(["score", *args, "--scorer", str(tiny_models / "ce"), "--device", "cpu", "--chunk-size", "2", "--batch", "2"])
    scores = pd.read_parquet(tmp_path / "scores" / "t.parquet")
    assert len(scores) == 5 and all(len(s) == 3 for s in scores["nanojev"]) and (scores["tokens"] > 0).all()
    (tmp_path / "ex.txt").write_text("q0\nq1\n")
    pools.main(["recall", *args, "--exclude", str(tmp_path / "ex.txt")])
    report = json.loads((tmp_path / "recall-t.json").read_text())
    assert report["subsets"]["clean"]["n"] == 3
    assert report["subsets"]["all"]["bm25"]["2"]["all_gold"] == 0.0  # BM25 ranks the distractor second
    assert report["subsets"]["all"]["bm25"]["3"]["all_gold"] == 1.0


def test_entropy_recorder_matches_raw_logits(tiny_models):
    import torch

    from stair.pool_answer import EntropyRecorder, Generator

    gen = Generator(str(tiny_models / "chat-lm"), "cpu", dtype="float32")
    enc = gen.tokenizer([gen.chat("sys", "Who directed the film?")], return_tensors="pt", add_special_tokens=False)
    rec = EntropyRecorder()
    out = gen.model.generate(**enc, max_new_tokens=4, do_sample=False, logits_processor=[rec], output_logits=True,
                             return_dict_in_generate=True, pad_token_id=gen.tokenizer.pad_token_id)
    logp = torch.stack([torch.log_softmax(s.float(), dim=-1) for s in out.logits], dim=1)
    assert torch.allclose(torch.stack(rec.steps, dim=1), -(logp.exp() * logp).sum(-1), atol=1e-5)


def test_pool_answer_end_to_end(tiny_models, tmp_path):
    from stair import pool_answer

    out = tmp_path / "exp8"
    (out / "pools").mkdir(parents=True)
    (out / "scores").mkdir()
    rows = [{"id": f"q{i}", "question": "Who directed the film?", "answer": "Kalo",
             "titles": ["Film", "Other", "Kalo"], "texts": ["It was directed by Kalo.", "Unrelated.", "Kalo was born."],
             "gold_idx": [0, 2]} for i in range(5)]
    pd.DataFrame(rows).to_parquet(out / "pools" / "test.parquet")
    pd.DataFrame({"id": [r["id"] for r in rows], "nanojev": [[0.9, 0.1, 0.8]] * 5, "tokens": [300] * 5}).to_parquet(
        out / "scores" / "test.parquet")
    lm = tiny_models / "chat-lm"
    plan = tmp_path / "plan.toml"
    plan.write_text(f'out = "{out}"\nseed = 0\n'
                    f'[[cells]]\nname = "one"\nmodel = "{lm}"\nparams = 1.0\ntop_k = 2\nsamples = 1\ntemperature = 0.3\n'
                    f'max_new_tokens = 4\nbatch_size = 2\n'
                    f'[[cells]]\nname = "fuse"\nmodel = "{lm}"\nparams = 1.0\ntop_k = 2\nsamples = 3\ntemperature = 0.9\n'
                    f'fuse = true\nmax_new_tokens = 4\nfuse_max_new_tokens = 4\nbatch_size = 2\n')
    args = ["--plan", str(plan), "--split", "test", "--device", "cpu", "--chunk-size", "2"]
    pool_answer.main(["run", *args])
    one, fuse = (pd.read_parquet(out / "answers" / "test" / f"{n}.parquet") for n in ("one", "fuse"))
    assert len(one) == len(fuse) == 5
    assert one["samples"].isna().all() and fuse["samples"].map(len).eq(3).all()
    assert (fuse["fuse_prompt_tokens"] > fuse["prompt_tokens"]).all()  # the fuser reads the prompt twice + answers
    nanojev = 2 * pool_answer.NANOJEV_PARAMS_B * 300
    tokens = 3 * fuse["prompt_tokens"] + fuse["gen_tokens"] + fuse["fuse_prompt_tokens"] + fuse["fuse_gen_tokens"]
    assert np.allclose(fuse["gflops"], 2 * 1.0 * tokens + nanojev)
    first = pd.read_parquet(out / "answers" / "test" / "one.parquet")
    pool_answer.main(["run", *args])  # resumes: nothing recomputed
    pd.testing.assert_frame_equal(first, pd.read_parquet(out / "answers" / "test" / "one.parquet"))
    pool_answer.main(["summary", "--plan", str(plan)])
    summary = pd.read_csv(out / "answers" / "summary.csv")
    assert set(summary["cell"]) == {"one", "fuse", "oracle"}
