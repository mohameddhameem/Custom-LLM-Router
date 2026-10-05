"""Evaluate routing policies on cached expert outputs.

For each policy, questions are escalated to the large expert in order of the policy's score,
at escalation rates from 0 to 1. Each rate gives one (mean cost, mean F1, mean EM) point.

Pre-generation policies (routers, oracle, random) pay for one expert per question. The
entropy baseline decides after the small expert has answered, so escalated questions pay for
both experts.

Cost is in one of three units (--cost): `flops` (2 x parameters x tokens, in GFLOPs, so a 7B token
costs more than a 1.5B one), `tokens` (prompt + generated, model size ignored) or `seconds`
(measured wall-clock per question, averaged over each chunk). Parameter counts come from the run's
config.toml: an expert's `params` (billions), else the size in its model name.
"""

import argparse
import json
import logging
import pickle
import re
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from groupproject.router import LABELS, escalation_label, predict_escalation

log = logging.getLogger(__name__)

RATES = np.round(np.linspace(0, 1, 21), 2)
COST_UNITS = ("flops", "tokens", "seconds")


def expert_params(cfg: dict) -> dict[str, float | None]:
    """Parameters in billions per expert: `params` if set, else parsed from e.g. "Qwen2.5-7B-Instruct"."""
    params = {}
    for expert in cfg["experts"]:
        if "params" in expert:
            params[expert["name"]] = float(expert["params"])
        else:
            m = re.search(r"(\d+(?:\.\d+)?)[Bb]\b", expert.get("model", ""))
            params[expert["name"]] = float(m.group(1)) if m else None
    return params


def with_cost_unit(df: pd.DataFrame, unit: str, params: dict[str, float | None]) -> pd.DataFrame:
    """Copy of df whose small_cost/large_cost are in `unit` (cached costs are tokens)."""
    df = df.copy()
    for name in ("small", "large"):
        if unit == "flops":
            if params.get(name) is None:
                raise ValueError(f"no parameter count for expert '{name}': add `params = <billions>` to its "
                                 "config or use --cost tokens")
            df[f"{name}_cost"] = 2 * params[name] * df[f"{name}_cost"]
        elif unit == "seconds":
            df[f"{name}_cost"] = df[f"{name}_seconds"]
    return df


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

    curves = {
        "random": random_curve(test),
        "oracle": [outcome(test, np.zeros(n), False), outcome(test, oracle, False), outcome(test, np.ones(n), False)],
        "small-entropy (post)": curve(test, test["small_uncertainty"].to_numpy(), post=True),
    }
    report = {
        "n_test": n,
        "tau": tau,
        "escalate_label_rate": {label: float(escalation_label(test, tau, label).mean()) for label in LABELS},
        "gold_pair_in_top2": float(test["gold_pair_in_top2"].mean()),
        "always_small": outcome(test, np.zeros(n), False),
        "always_large": outcome(test, np.ones(n), False),
        "oracle": outcome(test, oracle, False),
        "routers": {},
    }
    for name, router in routers.items():
        scores = predict_escalation(router, test)
        curves[f"router:{name}"] = curve(test, scores)
        label = router.get("label", "small-fails")  # routers.pkl from before labels were named
        entry = {"label": label, "ece": ece(scores, escalation_label(test, tau, label))}
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
    parser.add_argument("--cost", choices=COST_UNITS, default="flops", help="cost unit (see module docstring)")
    parser.add_argument("--exclude", type=Path, action="append", default=[],
                        help="file with test question ids to drop (e.g. ones nano-jev was tuned on); repeatable. "
                             "Writes report-clean.json and curves-clean.csv")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    saved = pickle.loads((args.run / "routers.pkl").read_bytes())
    params = expert_params(tomllib.loads((args.run / "config.toml").read_text()))
    test = pd.read_parquet(args.run / "test.parquet")
    excluded = set()
    for path in args.exclude:
        excluded |= {line.strip() for line in path.read_text().splitlines() if line.strip()}
    if excluded:
        before = len(test)
        test = test[~test["id"].isin(excluded)].reset_index(drop=True)
        log.info(f"excluded {before - len(test)} of {before} test questions")
    calib_path = args.run / "calib.parquet"
    calib = pd.read_parquet(calib_path) if calib_path.exists() else None

    single = {}
    for unit in COST_UNITS:
        try:
            t = with_cost_unit(test, unit, params)
        except ValueError:
            continue
        single[unit] = {"small": float(t["small_cost"].mean()), "large": float(t["large_cost"].mean())}
    test = with_cost_unit(test, args.cost, params)
    calib = with_cost_unit(calib, args.cost, params) if calib is not None else None

    report, curves = evaluate(test, calib, saved["routers"], saved["tau"], args.max_drop)
    report = {"cost_unit": args.cost, "expert_params_b": params, "excluded_files": [str(p) for p in args.exclude],
              **report, "single_expert_costs": single}
    suffix = "-clean" if excluded else ""
    (args.run / f"report{suffix}.json").write_text(json.dumps(report, indent=1))
    curves.to_csv(args.run / f"curves{suffix}.csv", index=False)

    rates = " | ".join(f"{k} {v:.3f}" for k, v in report["escalate_label_rate"].items())
    log.info(f"test: {report['n_test']} questions | escalate label rate {rates} | gold pair in top-2 {report['gold_pair_in_top2']:.3f}")
    log.info(f"cost unit: {args.cost}")
    for name in ("always_small", "always_large", "oracle"):
        p = report[name]
        log.info(f"  {name:<28} F1 {p['f1']:.3f} EM {p['em']:.3f} cost {p['cost']:.0f} rate {p['rate']:.2f}")
    for name, value in report["aiq"].items():
        log.info(f"  AIQ {name:<24} {value:.4f}")
    for name, entry in report["routers"].items():
        op = entry.get("operating_point")
        if op:
            log.info(f"  op {name:<25} F1 {op['f1']:.3f} cost {op['cost']:.0f} rate {op['rate']:.2f} | ECE {entry['ece']:.3f}")
    log.info(f"Saved {args.run / f'report{suffix}.json'} and {args.run / f'curves{suffix}.csv'}")


if __name__ == "__main__":
    main()
