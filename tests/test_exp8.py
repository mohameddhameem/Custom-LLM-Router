import json

import numpy as np
import pandas as pd
import pytest

from stair import exp8_eval as ev


def test_cheapest_correct_label():
    cost = np.tile([1.0, 5.0, 10.0], (5, 1))
    ok = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool)
    assert ev.cheapest_correct_label(ok, cost).tolist() == [0, 1, 2, 0, 0]  # none correct: the cheapest option
    cost[1] = [1.0, 12.0, 10.0]  # the label follows the cost, not the option's position
    assert ev.cheapest_correct_label(ok, cost)[1] == 2


def test_cumulative_choice_walks_up_the_options():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
    assert ev.cumulative_choice(p, 0.0).tolist() == [0, 0]
    assert ev.cumulative_choice(p, 0.5).tolist() == [0, 2]
    assert ev.cumulative_choice(p, 0.9).tolist() == [1, 2]
    assert ev.cumulative_choice(p, 1.0).tolist() == [2, 2]  # never past the last option


def test_judge_curve_cost_never_falls_as_t_rises():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([1, 1, 1], 200)
    ok = rng.random((200, 3)) < [0.4, 0.6, 0.8]
    cost = np.tile([3.0, 13.0, 120.0], (200, 1))
    pts = ev.judge_points(ok, cost, p)
    costs = [q["cost"] for q in pts]
    assert costs == sorted(costs) and len(pts) == len(ev.THRESHOLDS)
    assert pts[0]["cost"] == pytest.approx(3.0)  # t = 0: the cheapest option everywhere
    assert ev.judge_points(ok, cost, p, extra=2.0)[0]["cost"] == pytest.approx(5.0)


def test_score_points_between_two_options():
    ok = np.array([[0, 1], [1, 1], [0, 1], [1, 0]], dtype=bool)
    cost = np.tile([2.0, 10.0], (4, 1))
    score = np.array([0.9, 0.1, 0.8, 0.2])
    first, last = ev.score_points(ok, cost, score, 0, 1, post=False)[0], ev.score_points(ok, cost, score, 0, 1, post=False)[-1]
    assert (first["f1"], first["cost"]) == (0.5, 2.0) and (last["f1"], last["cost"]) == (0.75, 10.0)
    assert ev.score_points(ok, cost, score, 0, 1, post=True)[-1]["cost"] == 12.0  # post: escalated pay both calls (2 + 10)
    half = [p for p in ev.score_points(ok, cost, score, 0, 1, post=False) if p["rate"] == 0.5][0]
    assert half["f1"] == 1.0  # the two highest scores are the ones the strong option fixes


def test_operating_points():
    pts = [{"f1": 0.5, "cost": 1.0}, {"f1": 0.6, "cost": 5.0}, {"f1": 0.7, "cost": 20.0}, {"f1": 0.71, "cost": 90.0}]
    assert ev.choose_threshold(pts, budget=10.0) == pts[1]
    assert ev.choose_threshold(pts, budget=0.5) == pts[0]  # nothing fits: the cheapest point
    assert ev.choose_threshold(pts, floor=0.70) == pts[2]
    assert ev.choose_threshold(pts, floor=0.99) == pts[3]  # unreachable: the most accurate point


def test_bootstrap_pairs_policies():
    rng = np.random.default_rng(1)
    ok = rng.random((100, 2)) < [0.3, 0.8]
    cost = np.tile([1.0, 10.0], (100, 1))
    better = lambda ix: ev.always_points(ok[ix], cost[ix])
    boot = ev.bootstrap({"a": better, "b": better}, 100, 30, 0, (1.0, 10.0))
    assert ev.diff(boot, "a", "b")["p_le_0"] == 1.0  # identical policies: never better
    assert boot["aiq_ci95"]["a"][0] <= boot["aiq_ci95"]["a"][1]
    again = ev.bootstrap({"a": better}, 100, 30, 0, (1.0, 10.0))
    assert again["aiq_ci95"]["a"] == boot["aiq_ci95"]["a"]  # seeded


def test_evidence_features_and_logistic():
    f = ev.evidence_features(list(np.linspace(1, 0, 100)))
    assert f.shape == (7,) and f[0] > f[1] > f[2] and f[3] == pytest.approx(f[0] - f[1])
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = (X[:, 0] > 0).astype(int)
    model, C = ev.fit_logistic(X, y, 0)
    assert model.predict_proba(X)[:, 1][y == 1].mean() > 0.7 and C in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
