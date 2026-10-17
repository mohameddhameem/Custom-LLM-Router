"""Write a judge plan's test results in the team's shared layout (stair_common.schema), one row per question,
policy and seed, and validate them.

    export-common --plan configs/system-one-common.toml --out runs/common-judges --csv results/common/systemOne-hotpotqa-test.csv

Policies: always-small, always-large, and every finished judge at its operating point, per seed. The operating
point is the plan's: the cheapest threshold whose calib score (the plan's metric) is within max_drop of always-large.
Each row is re-scored with the standard ruler and official EM/F1 on the answer actually returned, and costed in
AgentTTS units:
- `units_retrieve` is nano-jev ranking the paragraphs (paid on every question);
- `units_answer` is the expert answers (both, for a post-answer judge that escalates);
- `units_router` is the judge itself.
"""

import argparse
import logging
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from stair_common.cost import UNIT_GFLOPS
from stair_common.schema import COLUMNS, validate
from stair_common.score import score

from stair.evaluate_routing import expert_params, pick_threshold, with_cost_unit
from stair.judge import INPUTS, judge_columns, judge_params_b, load_plan
from stair.metrics import metric_view

log = logging.getLogger(__name__)
TRACK = "systemOne"


def models_field(cfg: dict) -> str:
    parts = [f"retriever={cfg['scorer']['path']}"] if cfg.get("scorer", {}).get("path") else []
    parts += [f"{e['name']}={e['model']}" for e in cfg["experts"]]
    return ";".join(parts)


def rows_for(test: pd.DataFrame, escalate: np.ndarray, post: bool, policy: str, seed: int, router_units: np.ndarray,
             retrieve_units: np.ndarray, models: str) -> list[dict]:
    esc = escalate.astype(bool)
    answer_units = np.where(esc, test["large_cost"], test["small_cost"]) + (np.where(esc, test["small_cost"], 0) if post else 0)
    out = []
    for i, (qid, gold) in enumerate(zip(test["id"], test["answer"])):
        answer = str(test["large_pred"].iat[i] if esc[i] else test["small_pred"].iat[i] or "")
        parts = {"units_retrieve": float(retrieve_units[i]), "units_answer": float(answer_units[i]), "units_fuse": 0.0,
                 "units_router": float(router_units[i])}
        out.append({"dataset": "hotpotqa", "split": "test", "id": qid, "track": TRACK, "policy": policy, "menu": "B",
                    "seed": seed, "answer": answer, **score(answer, [str(gold)]), "units": sum(parts.values()), **parts,
                    "ms_router": None, "models": models})
    return out


def export(plan: dict, out: Path, pair: str = "A") -> pd.DataFrame:
    run = Path(plan["pairs"][pair])
    cfg = tomllib.loads((run / "config.toml").read_text())
    params, metric = expert_params(cfg), plan.get("metric", "f1")
    data = {}
    for split in ("calib", "test"):
        df = with_cost_unit(metric_view(pd.read_parquet(run / f"{split}.parquet"), metric), "agenttts", params)
        cols, specs = judge_columns(out, pair, split, plan)
        data[split] = df.merge(cols, on="id", how="left", validate="1:1") if len(cols) else df
    calib, test = data["calib"], data["test"]
    n, models = len(test), models_field(cfg)
    scorer_params = cfg.get("scorer", {}).get("params") or 0.0
    retrieve = (2 * scorer_params * test["scorer_tokens"] / UNIT_GFLOPS).to_numpy() if "scorer_tokens" in test else np.zeros(n)
    zero = np.zeros(n)
    rows = rows_for(test, zero, False, "always-small", 0, zero, retrieve, models)
    rows += rows_for(test, np.ones(n), False, "always-large", 0, zero, retrieve, models)
    for col, spec in specs.items():
        if spec.get("zero_shot"):
            continue
        post = INPUTS[spec["inputs"]]
        tokens = test.get(f"{col}_tokens")
        router = (2 * judge_params_b(spec["init"]) * tokens / UNIT_GFLOPS).to_numpy() if tokens is not None else zero
        for seed_col in sorted(c for c in test.columns if c.startswith(f"{col}#")):
            seed = int(seed_col.rsplit("#seed", 1)[1])
            t = pick_threshold(calib, calib[seed_col].to_numpy(), plan.get("max_drop", 0.01))
            esc = test[seed_col].to_numpy() >= t
            rows += rows_for(test, esc, post, col.removeprefix("judge/"), seed, router, retrieve, models)
    return pd.DataFrame(rows, columns=COLUMNS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="the judge run's --out (judges/<name>/seed<k>/)")
    parser.add_argument("--pair", default="A")
    parser.add_argument("--csv", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = export(load_plan(args.plan), args.out, args.pair)
    validate(df)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    summary = df.groupby(["policy", "seed"])[["lenient", "em", "f1", "units"]].mean().groupby("policy").mean()
    log.info(f"{len(df)} rows -> {args.csv}\n{summary.round(3).to_string()}")


if __name__ == "__main__":
    main()
