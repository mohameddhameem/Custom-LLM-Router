import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from groupproject import evaluate_routing, make_splits, router, run_experts
from groupproject.evaluate_routing import aiq, escalate_top, expert_params, outcome, with_cost_unit
from groupproject.evidence import passages_state
from groupproject.router import escalation_label
from groupproject.metrics import exact_match, f1_score
from synthetic import make_split

CONFIGS = Path(__file__).parent.parent / "configs"


@pytest.mark.parametrize("pred,gold,f1", [
    ("The Beatles", "beatles", 1.0),
    ("yes", "no", 0.0),
    ("yes", "yes.", 1.0),
    ("no", "no answer here", 0.0),  # yes/no never gets partial credit
    ("New York City", "New York", 0.8),
    ("Paris", "London", 0.0),
])
def test_f1_matches_official_rules(pred, gold, f1):
    assert f1_score(pred, gold) == pytest.approx(f1)


def test_exact_match_normalises():
    assert exact_match("  The  Who!", "who") == 1.0


def test_escalate_top_counts_and_order():
    esc = escalate_top(np.array([0.1, 0.9, 0.5, 0.7]), 0.5)
    assert esc.tolist() == [False, True, False, True]


def test_post_generation_policy_pays_for_both_experts():
    df = pd.DataFrame({"small_f1": [0.0], "large_f1": [1.0], "small_em": [0.0], "large_em": [1.0],
                       "small_cost": [10], "large_cost": [100]})
    assert outcome(df, np.array([True]), post=False)["cost"] == 100
    assert outcome(df, np.array([True]), post=True)["cost"] == 110


def test_escalation_labels():
    df = pd.DataFrame({"small_f1": [1.0, 0.0, 0.0, 0.5], "large_f1": [0.0, 1.0, 0.0, 0.9]})
    assert escalation_label(df, 0.8, "small-fails").tolist() == [0, 1, 1, 1]
    assert escalation_label(df, 0.8, "large-helps").tolist() == [0, 1, 0, 1]  # both wrong: not worth escalating


def test_expert_params_from_config_or_model_name():
    cfg = {"experts": [
        {"name": "small", "model": "Qwen/Qwen2.5-1.5B-Instruct"},
        {"name": "large", "model": "Qwen/Qwen2.5-7B-Instruct-AWQ", "params": 7.62},
    ]}
    assert expert_params(cfg) == {"small": 1.5, "large": 7.62}
    assert expert_params({"experts": [{"name": "small", "kind": "heuristic"}]}) == {"small": None}


def test_cost_units():
    df = pd.DataFrame({"small_cost": [100], "large_cost": [500], "small_seconds": [0.1], "large_seconds": [0.9]})
    params = {"small": 1.5, "large": 7.0}
    assert with_cost_unit(df, "tokens", params)["large_cost"].item() == 500
    flops = with_cost_unit(df, "flops", params)
    assert flops["small_cost"].item() == 300 and flops["large_cost"].item() == 7000
    assert with_cost_unit(df, "seconds", params)["large_cost"].item() == 0.9
    assert df["large_cost"].item() == 500  # input untouched
    with pytest.raises(ValueError, match="params"):
        with_cost_unit(df, "flops", {"small": 1.5, "large": None})


def test_passages_state_matches_nanojev_format():
    assert passages_state(["A: one.", "B: two."]) == "[1] A: one.\n[2] B: two."


def test_aiq_of_straight_line_is_its_mean():
    pts = [{"cost": 0.0, "f1": 0.2}, {"cost": 1.0, "f1": 0.6}]
    assert aiq(pts, 0.0, 1.0) == pytest.approx(0.4)


def test_aiq_uses_upper_hull_and_is_non_decreasing():
    pts = [{"cost": 0.0, "f1": 0.2}, {"cost": 0.5, "f1": 0.8}, {"cost": 0.6, "f1": 0.3}, {"cost": 1.0, "f1": 0.6}]
    assert aiq(pts, 0.0, 1.0) == pytest.approx(aiq(pts[:2] + [{"cost": 1.0, "f1": 0.8}], 0.0, 1.0))


def test_end_to_end_smoke(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    make_split(300, seed=0, level="medium").to_parquet(data / "train.parquet")
    make_split(120, seed=1, start=10_000).to_parquet(data / "validation.parquet")
    run = tmp_path / "run"

    make_splits.main(["--train", str(data / "train.parquet"), "--router-size", "200", "--calib-size", "80",
                      "--out", str(data / "splits.json")])
    common = ["--config", str(CONFIGS / "smoke.toml"), "--run", str(run)]
    for split in ("router_train", "calib"):
        run_experts.main(common + ["--data", str(data / "train.parquet"), "--splits", str(data / "splits.json"),
                                   "--split", split, "--name", split])
    run_experts.main(common + ["--data", str(data / "validation.parquet"), "--name", "test"])
    router.main(["--run", str(run), "--tau", "0.8"])
    evaluate_routing.main(["--run", str(run)])

    report = json.loads((run / "report.json").read_text())
    assert report["n_test"] == 120
    assert report["cost_unit"] == "flops"
    assert report["expert_params_b"] == {"small": 1.5, "large": 7.0}
    assert set(report["single_expert_costs"]) == {"flops", "tokens", "seconds"}
    rates = report["escalate_label_rate"]
    assert 0 < rates["large-helps"] <= rates["small-fails"] < 1
    small, large, oracle = (report[k]["f1"] for k in ("always_small", "always_large", "oracle"))
    assert large > small  # the experts must differ, or routing is not being exercised
    assert oracle >= large
    assert report["oracle"]["cost"] < report["always_large"]["cost"]
    assert set(report["routers"]) == {f"{i}/{label}" for i in ("question", "question+evidence")
                                      for label in ("small-fails", "large-helps")}
    for name, value in report["aiq"].items():
        assert 0 <= value <= 1, name
    # synthetic questions give away which expert succeeds, so a working router must beat random
    assert report["aiq"]["router:question/large-helps"] > report["aiq"]["random"]
    for entry in report["routers"].values():
        assert 0 <= entry["operating_point"]["rate"] <= 1
    curves = pd.read_csv(run / "curves.csv")
    assert len(curves[curves["policy"] == "router:question/large-helps"]) == len(evaluate_routing.RATES)

    excluded = pd.read_parquet(run / "test.parquet")["id"].head(20)
    (tmp_path / "exclude.txt").write_text("\n".join(excluded))
    evaluate_routing.main(["--run", str(run), "--exclude", str(tmp_path / "exclude.txt"), "--cost", "tokens"])
    clean = json.loads((run / "report-clean.json").read_text())
    assert clean["n_test"] == 100 and clean["cost_unit"] == "tokens"
    assert json.loads((run / "report.json").read_text())["n_test"] == 120  # full report untouched
