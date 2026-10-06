import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from groupproject import evaluate_routing, make_splits, router, run_experts, tau_sweep
from groupproject.evaluate_routing import aiq, as_arrays, curve, escalate_top, expert_params, outcome, with_cost_unit
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


@pytest.mark.parametrize("post", [False, True])
def test_curve_matches_escalating_the_top_scores(post):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.random(37) for c in evaluate_routing.CURVE_COLUMNS})
    scores = np.round(rng.random(37), 1)  # ties
    expected = [outcome(df, escalate_top(scores, r), post) for r in evaluate_routing.RATES]
    for got in (curve(df, scores, post), curve(as_arrays(df), scores, post)):
        assert [g["rate"] for g in got] == pytest.approx([e["rate"] for e in expected])
        for m in ("cost", "f1", "em"):
            assert [g[m] for g in got] == pytest.approx([e[m] for e in expected])


def test_small_cues_from_the_small_answer():
    df = pd.DataFrame({"question": ["Is it?", "Who?"], "small_pred": ["Yes", "Kalo"],
                       "small_token_entropy": [[0.5, 0.1], []], "small_generated_tokens": [2, 0],
                       "small_uncertainty": [0.3, np.inf]})
    x = router.add_features(df)
    assert x["small_says_yesno"].tolist() == [1.0, 0.0]
    assert x.loc[0, "small_entropy_max"] == 0.5 and x.loc[0, "small_entropy_first"] == 0.5
    assert x.loc[1, ["small_entropy_max", "small_uncertainty"]].isna().all()  # no tokens: missing, not infinite
    assert router.add_features(df[["question"]])[router.SMALL_COLUMNS].isna().all().all()


def test_bootstrap_intervals():
    rng = np.random.default_rng(1)
    n = 400
    small_f1 = (rng.random(n) < 0.5).astype(float)
    df = pd.DataFrame({"small_f1": small_f1, "large_f1": np.maximum(small_f1, rng.random(n) < 0.7),
                       "small_cost": 1.0, "large_cost": 5.0, "small_uncertainty": rng.random(n)})
    df["small_em"], df["large_em"] = df["small_f1"], df["large_f1"]
    a = as_arrays(df)
    good = df["large_f1"].to_numpy() - df["small_f1"].to_numpy() + 0.01 * rng.random(n)  # knows who wins
    scores = {"router:good": (good, False), "router:noise": (rng.random(n), False)}
    out = evaluate_routing.bootstrap(a, scores, {"router:good": (good > 0.5, False)}, 200, seed=0)
    assert out["aiq_diff"]["router:good - random"]["p_le_0"] == 0
    assert out["aiq_diff"]["router:good - random"]["ci95"][0] > 0
    # the hull lifts a noisy curve onto or above random's straight line, so noise never scores below it
    lo, hi = out["aiq_diff"]["router:noise - random"]["ci95"]
    assert 0 <= lo <= hi < out["aiq_diff"]["router:good - random"]["ci95"][0]
    assert out["op_f1_minus_large"]["router:good"]["ci95"] == pytest.approx([0, 0])  # escalates every large win
    full = evaluate_routing.policy_curves(a, scores)
    point = aiq(full["router:good"], 1.0, 5.0)
    assert out["aiq_ci95"]["router:good"][0] <= point <= out["aiq_ci95"]["router:good"][1]


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


def test_aiq_handles_points_at_the_same_cost():
    pts = [{"cost": 0.0, "f1": 0.2}, {"cost": 0.0, "f1": 0.4}, {"cost": 1.0, "f1": 0.6}]
    assert aiq(pts, 0.0, 1.0) == pytest.approx(0.5)


def test_temperature_scaling_recovers_overconfidence():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0, 2, 20_000)
    labels = (rng.random(20_000) < 1 / (1 + np.exp(-true_logits))).astype(int)
    overconfident = 1 / (1 + np.exp(-3 * true_logits))
    t = evaluate_routing.fit_temperature(overconfident, labels)
    assert t == pytest.approx(3, rel=0.1)
    scaled = evaluate_routing.scale(overconfident, t)
    assert evaluate_routing.ece(scaled, labels) < evaluate_routing.ece(overconfident, labels)
    assert (np.diff(scaled[np.argsort(overconfident)]) >= 0).all()  # monotonic: ranking and curves unchanged


def test_cpt_finds_cheapest_point_recovering_the_gap():
    pts = [{"rate": 0.0, "cost": 1.0, "f1": 0.4}, {"rate": 0.5, "cost": 2.0, "f1": 0.55},
           {"rate": 0.8, "cost": 3.0, "f1": 0.58}, {"rate": 1.0, "cost": 4.0, "f1": 0.6}]
    out = evaluate_routing.cpt(pts, 0.4, 0.6)
    assert out["50%"] == {"rate": 0.5, "cost": 2.0} and out["80%"] == {"rate": 0.8, "cost": 3.0}


def test_end_to_end_smoke(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    make_split(300, seed=0, level="medium").to_parquet(data / "train.parquet")
    make_split(120, seed=1, start=10_000).to_parquet(data / "validation.parquet")
    run = tmp_path / "run"

    make_splits.main(["--train", str(data / "train.parquet"), "--level", "medium", "--router-size", "200",
                      "--calib-size", "80", "--out", str(data / "splits.json")])
    common = ["--config", str(CONFIGS / "smoke.toml"), "--run", str(run)]
    for split in ("router_train", "calib"):
        run_experts.main(common + ["--data", str(data / "train.parquet"), "--splits", str(data / "splits.json"),
                                   "--split", split, "--name", split])
    run_experts.main(common + ["--data", str(data / "validation.parquet"), "--name", "test"])
    router.main(["--run", str(run), "--tau", "0.8"])
    evaluate_routing.main(["--run", str(run), "--bootstrap", "100"])

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
    assert set(report["routers"]) == {f"{i}/{label}" for i in router.ROUTER_INPUTS
                                      for label in ("small-fails", "large-helps")}
    assert report["routers"]["evidence+small/large-helps"]["post"] is True
    assert report["routers"]["evidence/large-helps"]["post"] is False
    boot = report["bootstrap"]
    assert boot["n"] == 100 and set(boot["aiq_ci95"]) == set(report["aiq"])
    assert len(boot["aiq_diff"]) == 2 * len(report["routers"])
    for name, (lo, hi) in boot["aiq_ci95"].items():
        assert lo <= hi, name
    for name, value in report["aiq"].items():
        assert 0 <= value <= 1, name
    # synthetic questions give away which expert succeeds, so a working router must beat random
    assert report["aiq"]["router:question/large-helps"] > report["aiq"]["random"]
    for entry in report["routers"].values():
        assert 0 <= entry["operating_point"]["rate"] <= 1
        assert entry["tuning"]["C"] in router.C_GRID and entry["temperature"] > 0
        assert set(entry["operating_point"]["by_answer_kind"]) == {"span", "yes/no"}
        assert sum(b["count"] for b in entry["reliability"]["scaled"]) == 120
    assert set(report["by_answer_kind"]) == {"always_small", "always_large"}
    assert report["cpt"]["oracle"]["80%"]["cost"] <= report["always_large"]["cost"]
    assert report["scorer_cost"]["tokens"] == 0  # the lexical scorer runs no model
    events = [json.loads(line)["event"] for line in (run / "provenance.jsonl").read_text().splitlines()]
    assert events.count("run-experts") == 3 and "train-router" in events and "eval-routing" in events
    curves = pd.read_csv(run / "curves.csv")
    assert len(curves[curves["policy"] == "router:question/large-helps"]) == len(evaluate_routing.RATES)

    excluded = pd.read_parquet(run / "test.parquet")["id"].head(20)
    (tmp_path / "exclude.txt").write_text("\n".join(excluded))
    evaluate_routing.main(["--run", str(run), "--exclude", str(tmp_path / "exclude.txt"), "--cost", "tokens"])
    clean = json.loads((run / "report-clean.json").read_text())
    assert clean["n_test"] == 100 and clean["cost_unit"] == "tokens"
    assert json.loads((run / "report.json").read_text())["n_test"] == 120  # full report untouched

    tau_sweep.main(["--run", str(run), "--tau", "0.5", "0.8"])
    sweep = pd.read_csv(run / "tau-sweep.csv")
    assert sorted(sweep["tau"].unique()) == [0.5, 0.8] and len(sweep) == 2 * len(report["routers"])
    at_08 = sweep[sweep["tau"] == 0.8].set_index("router")["aiq"]
    for name, value in at_08.items():  # same data, tau and seed: the sweep reproduces train-router + eval-routing
        assert value == pytest.approx(report["aiq"][f"router:{name}"]), name


def test_stage_settings_decide_what_can_be_reused():
    cfg = {"scorer": {"kind": "cross-encoder", "path": "a"},
           "experts": [{"name": "small", "model": "s", "top_k": 2, "batch_size": 32}, {"name": "large", "model": "l"}]}
    other_scorer = {**cfg, "scorer": {"kind": "cross-encoder", "path": "b"}}
    faster = {**cfg, "experts": [{**cfg["experts"][0], "batch_size": 8}, cfg["experts"][1]]}
    assert run_experts.stage_settings(cfg, "small") == run_experts.stage_settings(faster, "small")  # runtime only
    assert run_experts.stage_settings(cfg, "small") != run_experts.stage_settings(other_scorer, "small")  # reads top-k
    assert run_experts.stage_settings(cfg, "large") == run_experts.stage_settings(other_scorer, "large")  # all paragraphs
    assert run_experts.stage_settings(cfg, "missing") is None


def test_reuse_from_copies_only_identical_stages(tmp_path):
    make_split(60, seed=4).to_parquet(tmp_path / "val.parquet")
    variant = tmp_path / "variant.toml"
    variant.write_text((CONFIGS / "smoke.toml").read_text().replace("hops = 1", "hops = 0"))  # large expert differs
    first, second = tmp_path / "first", tmp_path / "second"
    common = ["--data", str(tmp_path / "val.parquet"), "--name", "test", "--chunk-size", "25"]
    run_experts.main(["--config", str(CONFIGS / "smoke.toml"), "--run", str(first)] + common)
    for stage in run_experts.STAGES:
        run_experts.main(["--config", str(variant), "--run", str(second), "--stage", stage,
                          "--reuse-from", str(first)] + common)

    reused = [json.loads(line) for line in (second / "provenance.jsonl").read_text().splitlines()
              if json.loads(line)["event"] == "reuse"]
    assert [r["copied"] for r in reused] == [{"evidence": 3}, {"small": 3}, {}, {}]
    events = [json.loads(line) for line in (second / "provenance.jsonl").read_text().splitlines()]
    assert [e["stage"] for e in events if e["event"] == "load"] == ["large"]  # only the large expert ran
    a, b = pd.read_parquet(first / "test.parquet"), pd.read_parquet(second / "test.parquet")
    pd.testing.assert_frame_equal(a.filter(regex="^(ev_|small_)"), b.filter(regex="^(ev_|small_)"))
    assert len(b) == 60 and b["large_pred"].notna().all()
