"""Run the scorer and both experts over a set of questions and cache the results.

Writes `<run>/<name>.parquet`: one row per question with evidence features and, for each expert,
its prediction, EM, F1, cost and uncertainty. Everything downstream reads these caches.

Work is split into stages (evidence, small, large, merge) and saved in chunks under
`<run>/<name>.parts/`. Rerunning the same command skips finished chunks, so an interrupted
Colab session resumes where it stopped. On a GPU, run each stage as its own command so only
one model holds GPU memory at a time.
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
from groupproject.experts import answer_batch, make_expert
from groupproject.metrics import exact_match, f1_score

log = logging.getLogger(__name__)

STAGES = ["evidence", "small", "large", "merge"]


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


def paragraphs_of(row) -> list[str]:
    return [paragraph_text(t, s) for t, s in zip(row["context_titles"], row["context_sentences"])]


def evidence_row(row, scorer, sufficiency_k: int) -> dict:
    paragraphs = paragraphs_of(row)
    scores = scorer.score(row["question"], paragraphs)
    order = np.argsort(-scores, kind="stable")
    sufficiency = scorer.sufficiency(row["question"], [paragraphs[i] for i in order[:sufficiency_k]])
    top2_titles = {row["context_titles"][i] for i in order[:2]}
    return {
        "id": row["id"], "question": row["question"], "answer": row["answer"], "type": row["type"],
        "ranked": order.tolist(),
        "gold_pair_in_top2": set(row["gold_titles"]) <= top2_titles,  # analysis only, never a router feature
        **evidence_features(scores, sufficiency),
    }


def expert_rows(chunk: pd.DataFrame, evidence: pd.DataFrame, expert) -> pd.DataFrame:
    ranked = dict(zip(evidence["id"], evidence["ranked"]))
    contexts = []
    for _, row in chunk.iterrows():
        paragraphs = paragraphs_of(row)
        contexts.append([paragraphs[i] for i in ranked[row["id"]]][: expert.top_k] if expert.top_k else paragraphs)
    start = time.perf_counter()
    outputs = answer_batch(expert, chunk["question"].tolist(), contexts)
    seconds = (time.perf_counter() - start) / len(chunk)
    n = expert.name
    return pd.DataFrame([{
        "id": qid,
        f"{n}_pred": out.answer,
        f"{n}_em": exact_match(out.answer, gold),
        f"{n}_f1": f1_score(out.answer, gold),
        f"{n}_cost": out.cost,
        f"{n}_uncertainty": out.uncertainty,
        f"{n}_seconds": seconds,
    } for qid, gold, out in zip(chunk["id"], chunk["answer"], outputs)])


class Parts:
    """Chunked stage outputs under <run>/<name>.parts/, pinned to one question list."""

    def __init__(self, run: Path, name: str, ids: list[str], chunk_size: int):
        self.dir = run / f"{name}.parts"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.chunks = [ids[i: i + chunk_size] for i in range(0, len(ids), chunk_size)]
        manifest = self.dir / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text())["chunks"] != self.chunks:
                raise ValueError(f"{self.dir} was started with a different question list or chunk size; "
                                 "use the same --split/--limit/--chunk-size or a new --run")
        else:
            manifest.write_text(json.dumps({"chunks": self.chunks}))

    def path(self, stage: str, i: int) -> Path:
        return self.dir / f"{stage}-{i:05d}.parquet"

    def missing(self, stage: str) -> list[int]:
        return [i for i in range(len(self.chunks)) if not self.path(stage, i).exists()]

    def write(self, stage: str, i: int, df: pd.DataFrame) -> None:
        tmp = self.path(stage, i).with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(self.path(stage, i))  # atomic: a killed session never leaves a half-written chunk

    def read(self, stage: str) -> pd.DataFrame:
        missing = self.missing(stage)
        if missing:
            raise FileNotFoundError(f"stage '{stage}' has {len(missing)} unfinished chunks in {self.dir}")
        return pd.concat([pd.read_parquet(self.path(stage, i)) for i in range(len(self.chunks))], ignore_index=True)


def run_evidence(questions: pd.DataFrame, parts: Parts, cfg: dict) -> None:
    todo = parts.missing("evidence")
    if not todo:
        return
    scorer = make_scorer(cfg.get("scorer", {}))
    k = cfg.get("scorer", {}).get("sufficiency_top_k", 2)
    by_id = questions.set_index("id", drop=False)
    for i in todo:
        rows = [evidence_row(row, scorer, k) for _, row in by_id.loc[parts.chunks[i]].iterrows()]
        parts.write("evidence", i, pd.DataFrame(rows))
        log.info(f"  evidence chunk {i + 1}/{len(parts.chunks)}")
    if getattr(scorer, "pairs_seen", 0):
        log.info(f"  cross-encoder pairs truncated: {scorer.truncated}/{scorer.pairs_seen}")


def run_expert(questions: pd.DataFrame, parts: Parts, cfg: dict, name: str) -> None:
    todo = parts.missing(name)
    if not todo:
        return
    evidence = parts.read("evidence")
    expert = make_expert(next(e for e in cfg["experts"] if e["name"] == name))
    by_id = questions.set_index("id", drop=False)
    for i in todo:
        rows = expert_rows(by_id.loc[parts.chunks[i]], evidence, expert)
        parts.write(name, i, rows)
        log.info(f"  {name} chunk {i + 1}/{len(parts.chunks)}: {rows[f'{name}_seconds'].iloc[0]:.3f} s/question")


def merge(parts: Parts, out: Path) -> None:
    df = parts.read("evidence").drop(columns="ranked")
    for name in ("small", "large"):
        df = df.merge(parts.read(name), on="id", how="left", validate="one_to_one")
    for name in ("small", "large"):
        log.info(f"  {name}: EM {df[f'{name}_em'].mean():.3f} F1 {df[f'{name}_f1'].mean():.3f} "
                 f"cost {df[f'{name}_cost'].mean():.0f} sec/q {df[f'{name}_seconds'].mean():.3f}")
    log.info(f"  gold pair in top-2: {df['gold_pair_in_top2'].mean():.3f}")
    df.to_parquet(out, index=False)
    log.info(f"Saved {out}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="parquet from prepare-hotpotqa")
    parser.add_argument("--splits", type=Path, default=Path("data/hotpotqa/splits.json"))
    parser.add_argument("--split", help="key in --splits (router_train, calib); omit to use all rows")
    parser.add_argument("--limit", type=int, help="first N questions only")
    parser.add_argument("--run", type=Path, required=True, help="run directory")
    parser.add_argument("--name", required=True, help="cache name: router_train, calib or test")
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    parser.add_argument("--chunk-size", type=int, default=500)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    args.run.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, args.run / "config.toml")

    questions = load_questions(args.data, args.splits, args.split, args.limit)
    parts = Parts(args.run, args.name, questions["id"].tolist(), args.chunk_size)
    stages = STAGES if args.stage == "all" else [args.stage]
    log.info(f"{args.name}: {len(questions)} questions in {len(parts.chunks)} chunks | stages {stages}")
    for stage in stages:
        if stage == "evidence":
            run_evidence(questions, parts, cfg)
        elif stage == "merge":
            merge(parts, args.run / f"{args.name}.parquet")
        else:
            run_expert(questions, parts, cfg, stage)


if __name__ == "__main__":
    main()
