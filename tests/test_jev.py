"""The Jev novelty programme (docs/experiments/2026-10-24-jev-preregistration.md): label budgets, judge grids and the
nano-jev pretraining arms."""

import json
from pathlib import Path

import numpy as np
import pytest

from stair import judge, nanojev_pretrain

CONFIGS = Path(__file__).parent.parent / "configs"
FAKE_EXAMPLE = '\ndef _example(decision, label, state, source, **fields):\n    return {"decision": decision, "question": str(fields), "options": ["a", "b", "c"], "state": state,\n            "label": label, "source": source}\n'


def test_budgets_nest_and_keep_the_label_rate():
    y = (np.random.default_rng(0).random(1000) < 0.2).astype(int)
    order = judge.budget_order(y, seed=3)
    assert sorted(order) == list(range(1000))  # a permutation
    for b in (50, 100, 500):
        assert abs(y[order[:b]].mean() - y.mean()) <= 1.5 / b + 0.01  # every prefix is stratified
    assert set(order[:50]) <= set(order[:100])  # nested by construction
    assert not np.array_equal(order, judge.budget_order(y, seed=4))  # another seed, another draw


def test_a_budget_counts_its_dev_hold_out():
    y = (np.random.default_rng(1).random(400) < 0.3).astype(int)
    tr, dev = judge.budget_split(y, 50, 0.1, seed=0)
    assert len(tr) + len(dev) == 50 and len(dev) == 10  # at least 10 to pick the epoch
    assert set(tr) | set(dev) == set(judge.budget_order(y, 0)[:50])
    tr, dev = judge.budget_split(y, 400, 0.1, seed=0)
    assert len(dev) == 40
    with pytest.raises(ValueError, match="budget"):
        judge.budget_split(y, 401, 0.1, seed=0)


def test_a_judge_entry_expands_to_its_grid_and_names_stay_apart(tmp_path):
    spec = {"inputs": ["qp", "qpa"], "label": "large-helps", "init": ["sdmlai/nano-jev@v1.0", "runs/x/model"],
            "head": ["typed", "plain"], "budget": [50, 4814], "train_pair": "A", "seeds": [0, 1]}
    specs = judge.expand(spec)
    assert len(specs) == 16 and all(s["seeds"] == [0, 1] for s in specs)
    assert judge.judge_name({**specs[0], "head": "typed", "budget": None}) == "qp-large-helps-nanojev-A"
    named = {**specs[-1], "init_name": "nanojev-repro"}
    assert judge.judge_name(named) == f"{named['inputs']}-large-helps-nanojev-repro-A-plain-b4814"

    def plan(names: str) -> str:
        return f"""
tau = 0.5
seeds = [0]
[pairs]
A = "runs/a"
{names}
[[judges]]
inputs = "qp"
label = "large-helps"
init = ["runs/one/model", "runs/two/model"]
train_pair = "A"
"""

    path = tmp_path / "plan.toml"
    path.write_text(plan(""))
    with pytest.raises(ValueError, match="share a name"):  # both local inits would be called "model"
        judge.load_plan(path)
    path.write_text(plan('[init_names]\n"runs/one/model" = "one"\n"runs/two/model" = "two"'))
    assert {judge.judge_name(s) for s in judge.load_plan(path)["judges"]} == {"qp-large-helps-one-A", "qp-large-helps-two-A"}


def test_the_e1_plan_is_the_preregistered_grid():
    plan = judge.load_plan(CONFIGS / "e1-label-curve.toml")
    specs = plan["judges"]
    assert len(specs) == 2 * 5 * 2 * 7  # inputs x starting points x heads x budgets
    runs = sum(len(s["seeds"]) for s in specs)
    assert runs == 2 * 5 * 2 * (4 * 10 + 3 * 5)
    assert {s["budget"] for s in specs} == {50, 100, 250, 500, 1000, 2000, 4814}


def _fake_recipe(root):
    """A stand-in for nano-jev's `nanojev.data`: its two non-HotpotQA builders, tagged by source."""
    pkg = root / "nanojev"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "data.py").write_text('''
def _ex(source, split, i):
    return {"decision": "sufficient", "question": f"{source} {split} {i}", "options": ["yes", "no"],
            "state": "s", "label": i % 2, "source": source}

def squad2_examples(split, n, rng):
    return [_ex("squad2", split, rng.randint(0, 10**6)) for _ in range(n)]

def mnli_examples(split, n, rng):
    return [_ex("mnli", split, rng.randint(0, 10**6)) for _ in range(n)]
''')


class _Rows:
    """A stand-in for a SQuAD 2.0 split: column access by name, row access by index."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        return [r[key] for r in self.rows] if isinstance(key, str) else self.rows[key]


def _squad(n_titles=4, per_title=3):
    rows = []
    for t in range(n_titles):
        for p in range(per_title):
            context = f"paragraph {t}-{p} mentions answer{t}{p}"
            rows.append({"title": f"T_{t}", "context": context, "question": f"q{t}{p}?",
                         "answers": {"text": [f"answer{t}{p}"]}})
            rows.append({"title": f"T_{t}", "context": context, "question": f"u{t}{p}?", "answers": {"text": []}})
    return _Rows(rows)


def _example(decision, label, state, source, **fields):
    return {"decision": decision, "question": str(fields), "options": ["a", "b", "c"], "state": state,
            "label": label, "source": source}


def test_squad_relevance_mirrors_hotpotqa_four_examples():
    import random

    out = nanojev_pretrain.squad_relevance_examples(_squad(), 3, random.Random(0), _example)
    assert len(out) == 12 and [e["label"] for e in out[:4]] == [2, 1, 0, 0]
    gold, partial, other = out[0]["state"], out[1]["state"], out[2]["state"]
    assert gold.split(":")[0] == partial.split(":")[0] != other.split(":")[0]  # same article, then another one
    answer = gold.split("mentions ")[1]
    assert answer not in partial  # "partially relevant": same article, without the answer string


def test_the_nohotpot_arm_has_no_hotpotqa_anywhere(tmp_path, monkeypatch):
    import sys

    _fake_recipe(tmp_path / "src")
    data_py = tmp_path / "src" / "nanojev" / "data.py"
    data_py.write_text(data_py.read_text() + FAKE_EXAMPLE)
    monkeypatch.setattr(nanojev_pretrain, "NOHOTPOT", {"relevance_train": 2, "squad_train": 6, "mnli_train": 4,
                                                       "relevance_eval": 1, "squad_eval": 2, "mnli_eval": 3})
    monkeypatch.setattr(nanojev_pretrain, "load_squad", lambda split: _squad())
    monkeypatch.setattr("sys.path", list(sys.path))  # build_nohotpot prepends the recipe folder
    for mod in [m for m in sys.modules if m == "nanojev" or m.startswith("nanojev.")]:
        monkeypatch.delitem(sys.modules, mod)
    data = nanojev_pretrain.build_nohotpot(tmp_path / "src", seed=0)
    assert len(data["train"]) == 8 + 10 and len(data["calib"]) == 4 + 5 and len(data["test"]) == 4 + 5
    assert {ex["source"] for split in data.values() for ex in split} == {"squad2", "mnli"}
    for split, examples in data.items():
        nanojev_pretrain.write_jsonl(examples, tmp_path / "data" / f"{split}.jsonl")
    counts = nanojev_pretrain.source_counts(tmp_path / "data")
    assert counts["train"] == {"squad2/relevance": 8, "squad2/sufficient": 6, "mnli/sufficient": 4}
    again = nanojev_pretrain.build_nohotpot(tmp_path / "src", seed=0)
    assert json.dumps(again) == json.dumps(data)  # deterministic for a seed


def _musique_example(i: int) -> dict:
    paragraphs = [{"idx": k, "title": f"T{k}", "paragraph_text": f" text {i}-{k} ", "is_supporting": k in (3, 7)}
                  for k in range(20)]
    return {"id": f"2hop__{i}_{i + 1}", "paragraphs": paragraphs, "question": f"q{i}?", "answer": "A",
            "answer_aliases": ["A", "Alpha"], "answerable": True}


def test_musique_becomes_the_hotpotqa_schema():
    from stair.prepare_hotpotqa import flatten_example
    from stair.prepare_multihop import convert

    df = convert("musique", [_musique_example(1), _musique_example(2)])
    row = df.iloc[0]
    assert row["type"] == "2hop" and row["num_context_paragraphs"] == 20 and list(row["answers"]) == ["A", "Alpha"]
    assert list(row["context_sentences"][3]) == ["text 1-3"] and list(row["gold_titles"]) == ["T3", "T7"]
    hotpot_like = {"id": "x", "question": "q", "answer": "a", "type": "bridge", "level": "hard",
                   "context": {"title": ["A"], "sentences": [["s"]]}, "supporting_facts": {"title": ["A"], "sent_id": [0]}}
    assert set(flatten_example(hotpot_like)) | {"answers"} == set(df.columns)  # the same columns run-experts reads


def test_multihop_splits_follow_the_team_rule(tmp_path):
    import pandas as pd
    from stair_common.splits import shuffled

    from stair.make_splits import main as make_splits_main

    train = [f"t{i:05d}" for i in range(6000)]
    pd.DataFrame({"id": train}).to_parquet(tmp_path / "train.parquet")
    pd.DataFrame({"id": [f"d{i}" for i in range(300)]}).to_parquet(tmp_path / "dev.parquet")
    make_splits_main(["--dataset", "musique", "--data-dir", str(tmp_path)])
    splits = json.loads((tmp_path / "splits.json").read_text())
    assert splits["router_train"] == sorted(shuffled(train)[:4814]) and len(splits["calib"]) == 500
    assert not set(splits["router_train"]) & set(splits["calib"]) and len(splits["test"]) == 300


def test_a_dataset_without_gold_answers_is_refused():
    import pandas as pd

    from stair.prepare_multihop import SIZES, check

    frames = {part: pd.DataFrame({"id": [f"{part}{i}" for i in range(n)], "answer": ["a"] * n})
              for part, n in SIZES["2wikimultihopqa"].items()}
    check("2wikimultihopqa", frames)
    frames["dev"].loc[:, "answer"] = ""  # the team's 2Wiki source: the official test set, answers hidden
    with pytest.raises(ValueError, match="without a gold answer"):
        check("2wikimultihopqa", frames)


def test_twowiki_splits_cut_the_answered_dev_set(tmp_path):
    import pandas as pd
    from stair_common.splits import shuffled

    from stair.make_splits import main as make_splits_main

    pd.DataFrame({"id": [f"t{i}" for i in range(5000)]}).to_parquet(tmp_path / "train.parquet")
    dev = [f"d{i}" for i in range(4000)]
    pd.DataFrame({"id": dev}).to_parquet(tmp_path / "dev.parquet")
    make_splits_main(["--dataset", "2wikimultihopqa", "--data-dir", str(tmp_path)])
    splits = json.loads((tmp_path / "splits.json").read_text())
    order = shuffled(dev)
    assert splits["calib"] == sorted(order[:500]) and splits["test"] == sorted(order[500:1000])
    assert splits["test_ext"] == sorted(order[1000:3000])


def test_holm_and_le_auc():
    from stair.label_curve import holm, le_auc

    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": 0.03, "c": 0.06, "b": 0.06}
    assert le_auc({50: 0.6, 100: 0.6, 4814: 0.6}) == pytest.approx(0.6)  # flat curve: its value
    assert le_auc({64: 0.5, 128: 0.7}) == pytest.approx(0.6)  # the mean over log2(budget)


def test_label_curve_finds_a_pretraining_gain_that_fades_with_labels():
    from stair import label_curve as lc

    rng = np.random.default_rng(0)
    n = 600
    small = (rng.random(n) < 0.55).astype(float)
    large = np.maximum(small, (rng.random(n) < 0.35).astype(float))
    a = {"small_f1": small, "large_f1": large, "small_em": small, "large_em": large,
         "small_cost": np.full(n, 0.5), "large_cost": np.full(n, 10.0), "small_uncertainty": rng.random(n)}
    helps = large - small  # 1 where escalating fixes the answer

    def judge(quality: float, seed: int) -> np.ndarray:  # a score that sees `quality` of the truth
        r = np.random.default_rng(seed)
        return quality * helps + (1 - quality) * r.random(n)

    arms = {}
    for budget, gain in ((50, 0.3), (100, 0.25), (4814, 0.02)):
        for init, base in (("nanojev", 0.4 + gain), ("minilm-l12-h384-uncased", 0.4), ("nanojev-nohotpot", 0.4 + gain)):
            arms[(init, "typed", "qp", budget)] = {s: judge(base, 100 * budget + s) for s in range(3)}
        arms[("nanojev", "plain", "qp", budget)] = {s: judge(0.4, 7 + s) for s in range(3)}
    report = lc.analyse(a, arms, n_boot=30, seed=0)
    stats = report["stats"]
    assert stats["H1a/qp"]["estimate"] > 0 and stats["H1a/qp"]["p_le_0"] < 0.05
    assert stats["H1b/qp"]["estimate"] > 0  # the gain shrinks as labels grow
    assert stats["H1c/qp"]["estimate"] > 0  # noHotpot keeps the gain here
    assert stats["H1d/qp"]["estimate"] > 0 and "p_holm" in stats["H1d/qp"]
    assert {c["budget"] for c in report["cells"]} == {50, 100, 4814}
    assert all(0 < c["gap_closed"] < 1 for c in report["cells"])
