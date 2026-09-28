"""Evaluate routing policies on cached expert outputs.

For each policy, questions are escalated to the large expert in order of the policy's score,
at escalation rates from 0 to 1. Each rate gives one (mean cost, mean F1, mean EM) point.

Pre-generation policies (routers, oracle, random) pay for one expert per question. The
entropy baseline decides after the small expert has answered, so escalated questions pay for
both experts.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from groupproject.router import escalation_label, predict_escalation

log = logging.getLogger(__name__)

RATES = np.round(np.linspace(0, 1, 21), 2)


def outcome(df: pd.DataFrame, escalate: np.ndarray, post: bool) -> dict[str, float]:
    esc = escalate.astype(bool)
    f1 = np.where(esc, df["large_f1"], df["small_f1"])
    em = np.where(esc, df["large_em"], df["small_em"])
    cost = np.where(esc, df["large_cost"], df["small_cost"]).astype(float)
    if post:
        cost = cost + np.where(esc, df["small_cost"], 0)
    return {"rate": float(esc.mean()), "cost": float(cost.mean()), "f1": float(f1.mean()), "em": float(em.mean())}


def escalate_top(scores: np.ndarray, rate: float) -> np.ndarray:
    """Escalate the round(rate * n) highest-scoring questions (ties broken by position)."""
    n = round(rate * len(scores))
    esc = np.zeros(len(scores), dtype=bool)
    esc[np.argsort(-scores, kind="stable")[:n]] = True
    return esc


def curve(df: pd.DataFrame, scores: np.ndarray, post: bool = False) -> list[dict]:
    return [outcome(df, escalate_top(scores, r), post) for r in RATES]


def random_curve(df: pd.DataFrame) -> list[dict]:
    """Expected value of escalating a random fraction: a linear mix of the two experts."""
    s, l = outcome(df, np.zeros(len(df)), False), outcome(df, np.ones(len(df)), False)
    return [{k: (1 - r) * s[k] + r * l[k] for k in s} | {"rate": float(r)} for r in RATES]


def aiq(points: list[dict], cmin: float, cmax: float) -> float:
    """RouterBench AIQ: mean F1 under the non-decreasing upper convex hull over [cmin, cmax]."""
    pts = sorted({(p["cost"], p["f1"]) for p in points})
    hull: list[tuple[float, float]] = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    xs = np.array([h[0] for h in hull])
    ys = np.maximum.accumulate(np.array([h[1] for h in hull]))
    grid = np.linspace(cmin, cmax, 201)
    return float(np.interp(grid, xs, ys).mean())


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    idx = np.minimum((probs * bins).astype(int), bins - 1)
    return float(sum(
        abs(probs[idx == b].mean() - labels[idx == b].mean()) * (idx == b).mean()
        for b in range(bins) if (idx == b).any()
    ))


def pick_threshold(calib: pd.DataFrame, scores: np.ndarray, max_drop: float) -> float:
    """Highest score threshold whose calib F1 is within max_drop (relative) of always-large."""
    target = (1 - max_drop) * calib["large_f1"].mean()
    for t in sorted(set(scores), reverse=True) + [-np.inf]:
        if outcome(calib, scores >= t, False)["f1"] >= target:
            return float(t)
    return -np.inf


def by_type(df: pd.DataFrame, escalate: np.ndarray) -> dict:
    return {t: outcome(df[m], escalate[m.to_numpy()], False) for t, m in ((t, df["type"] == t) for t in sorted(df["type"].unique()))}


def evaluate(test: pd.DataFrame, calib: pd.DataFrame | None, routers: dict, tau: float, max_drop: float) -> tuple[dict, pd.DataFrame]:
    n = len(test)
    oracle = (test["large_f1"] > test["small_f1"]).to_numpy()
    cmin, cmax = test["small_cost"].mean(), test["large_cost"].mean()
    labels = escalation_label(test, tau)

    curves = {
        "random": random_curve(test),
        "oracle": [outcome(test, np.zeros(n), False), outcome(test, oracle, False), outcome(test, np.ones(n), False)],
        "small-entropy (post)": curve(test, test["small_uncertainty"].to_numpy(), post=True),
    }
    report = {
        "n_test": n,
        "tau": tau,
        "escalate_label_rate": float(labels.mean()),
        "gold_pair_in_top2": float(test["gold_pair_in_top2"].mean()),
        "always_small": outcome(test, np.zeros(n), False),
        "always_large": outcome(test, np.ones(n), False),
        "oracle": outcome(test, oracle, False),
        "routers": {},
    }
    for name, router in routers.items():
        scores = predict_escalation(router, test)
        curves[f"router:{name}"] = curve(test, scores)
        entry = {"ece": ece(scores, labels)}
        if calib is not None:
            t = pick_threshold(calib, predict_escalation(router, calib), max_drop)
            esc = scores >= t
            entry["operating_point"] = {"threshold": t, **outcome(test, esc, False), "by_type": by_type(test, esc)}
        report["routers"][name] = entry
    report["aiq"] = {name: aiq(pts, cmin, cmax) for name, pts in curves.items()}
    report["by_type"] = {"always_small": by_type(test, np.zeros(n, bool)), "always_large": by_type(test, np.ones(n, bool))}

    rows = [{"policy": name, **p} for name, pts in curves.items() for p in pts]
    return report, pd.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--max-drop", type=float, default=0.01, help="allowed relative F1 drop vs always-large at the operating point")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    saved = pickle.loads((args.run / "routers.pkl").read_bytes())
    test = pd.read_parquet(args.run / "test.parquet")
    calib_path = args.run / "calib.parquet"
    calib = pd.read_parquet(calib_path) if calib_path.exists() else None

    report, curves = evaluate(test, calib, saved["routers"], saved["tau"], args.max_drop)
    (args.run / "report.json").write_text(json.dumps(report, indent=1))
    curves.to_csv(args.run / "curves.csv", index=False)

    log.info(f"test: {report['n_test']} questions | escalate label rate {report['escalate_label_rate']:.3f} | gold pair in top-2 {report['gold_pair_in_top2']:.3f}")
    for name in ("always_small", "always_large", "oracle"):
        p = report[name]
        log.info(f"  {name:<28} F1 {p['f1']:.3f} EM {p['em']:.3f} cost {p['cost']:.0f} rate {p['rate']:.2f}")
    for name, value in report["aiq"].items():
        log.info(f"  AIQ {name:<24} {value:.4f}")
    for name, entry in report["routers"].items():
        op = entry.get("operating_point")
        if op:
            log.info(f"  op {name:<25} F1 {op['f1']:.3f} cost {op['cost']:.0f} rate {op['rate']:.2f} | ECE {entry['ece']:.3f}")
    log.info(f"Saved {args.run / 'report.json'} and {args.run / 'curves.csv'}")


if __name__ == "__main__":
    main()
