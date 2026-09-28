"""Run the scorer and both experts over a set of questions and cache the results.

Writes one row per question with evidence features and, for each expert, its prediction,
EM, F1, cost and uncertainty. Everything downstream reads these caches, so experts run once.
"""

import argparse
import json
import logging
import shutil
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from groupproject.evidence import evidence_features, make_scorer, paragraph_text
from groupproject.experts import make_expert
from groupproject.metrics import exact_match, f1_score

log = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    cfg = tomllib.loads(path.read_text())
    names = [e["name"] for e in cfg["experts"]]
    if names != ["small", "large"]:
        raise ValueError(f"config must define experts 'small' then 'large', got {names}")
    return cfg


def load_questions(data: Path, splits: Path | None, split: str | None, limit: int | None) -> pd.DataFrame:
    df = pd.read_parquet(data)
    if split:
        ids = set(json.loads(splits.read_text())[split])
        df = df[df["id"].isin(ids)]
    if limit:
        df = df.head(limit)
    return df.reset_index(drop=True)


def process_question(row, scorer, experts, sufficiency_k: int) -> dict:
    paragraphs = [paragraph_text(t, s) for t, s in zip(row["context_titles"], row["context_sentences"])]
    scores = scorer.score(row["question"], paragraphs)
    order = np.argsort(-scores, kind="stable")
    ranked = [paragraphs[i] for i in order]
    sufficiency = scorer.sufficiency(row["question"], ranked[:sufficiency_k])

    out = {"id": row["id"], "question": row["question"], "answer": row["answer"], "type": row["type"]}
    gold = set(row["gold_titles"])
    top2_titles = {row["context_titles"][i] for i in order[:2]}
    out["gold_pair_in_top2"] = gold <= top2_titles  # analysis only, never a router feature
    out.update(evidence_features(scores, sufficiency))

    for expert in experts:
        context = ranked[: expert.top_k] if expert.top_k else paragraphs
        start = time.perf_counter()
        res = expert.answer(row["question"], context)
        out[f"{expert.name}_seconds"] = time.perf_counter() - start
        out[f"{expert.name}_pred"] = res.answer
        out[f"{expert.name}_em"] = exact_match(res.answer, row["answer"])
        out[f"{expert.name}_f1"] = f1_score(res.answer, row["answer"])
        out[f"{expert.name}_cost"] = res.cost
        out[f"{expert.name}_uncertainty"] = res.uncertainty
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="parquet from prepare-hotpotqa")
    parser.add_argument("--splits", type=Path, default=Path("data/hotpotqa/splits.json"))
    parser.add_argument("--split", help="key in --splits (router_train, calib); omit to use all rows")
    parser.add_argument("--limit", type=int, help="first N questions only")
    parser.add_argument("--run", type=Path, required=True, help="run directory")
    parser.add_argument("--name", required=True, help="cache name: router_train, calib or test")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    args.run.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, args.run / "config.toml")

    questions = load_questions(args.data, args.splits, args.split, args.limit)
    log.info(f"{args.name}: {len(questions)} questions from {args.data}")
    scorer = make_scorer(cfg.get("scorer", {}))
    experts = [make_expert(e) for e in cfg["experts"]]
    k = cfg.get("scorer", {}).get("sufficiency_top_k", 2)

    rows = []
    for i, row in questions.iterrows():
        rows.append(process_question(row, scorer, experts, k))
        if (i + 1) % 50 == 0:
            log.info(f"  {i + 1}/{len(questions)}")
    df = pd.DataFrame(rows)

    for e in experts:
        log.info(f"  {e.name}: EM {df[f'{e.name}_em'].mean():.3f} F1 {df[f'{e.name}_f1'].mean():.3f} "
                 f"cost {df[f'{e.name}_cost'].mean():.0f} sec/q {df[f'{e.name}_seconds'].mean():.2f}")
    log.info(f"  gold pair in top-2: {df['gold_pair_in_top2'].mean():.3f}")
    if getattr(scorer, "pairs_seen", 0):
        log.info(f"  cross-encoder pairs truncated: {scorer.truncated}/{scorer.pairs_seen}")

    out = args.run / f"{args.name}.parquet"
    df.to_parquet(out, index=False)
    log.info(f"Saved {out}")


if __name__ == "__main__":
    main()
