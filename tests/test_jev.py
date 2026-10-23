"""The Jev novelty programme (docs/experiments/2026-10-24-jev-preregistration.md): label budgets, judge grids and the
nano-jev pretraining arms."""

import json
from pathlib import Path

import numpy as np
import pytest

from stair import judge, nanojev_pretrain

CONFIGS = Path(__file__).parent.parent / "configs"


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


def test_the_nohotpot_arm_has_no_hotpotqa_anywhere(tmp_path, monkeypatch):
    _fake_recipe(tmp_path / "src")
    monkeypatch.setattr(nanojev_pretrain, "NOHOTPOT", {"squad_train": 6, "mnli_train": 4, "squad_eval": 2, "mnli_eval": 3})
    monkeypatch.setattr("sys.path", list(__import__("sys").path))  # build_nohotpot prepends the recipe folder
    data = nanojev_pretrain.build_nohotpot(tmp_path / "src", seed=0)
    assert len(data["train"]) == 10 and len(data["calib"]) == 5 and len(data["test"]) == 5  # eval sets halved
    assert {ex["source"] for split in data.values() for ex in split} == {"squad2", "mnli"}
    for split, examples in data.items():
        nanojev_pretrain.write_jsonl(examples, tmp_path / "data" / f"{split}.jsonl")
    counts = nanojev_pretrain.source_counts(tmp_path / "data")
    assert counts["train"] == {"squad2/sufficient": 6, "mnli/sufficient": 4}
    again = nanojev_pretrain.build_nohotpot(tmp_path / "src", seed=0)
    assert json.dumps(again) == json.dumps(data)  # deterministic for a seed
