"""Escalation routers: predict whether a question should go to the large expert.

Routers with the same label share the model (logistic regression) and training data and differ
only in input: `question` sees question text and lexical cues; `question+evidence` also sees the
paragraph-score features computed before any expert runs.

Two labels are trained, because which one is right is part of the experiment:
`small-fails` (small F1 < tau) also escalates questions the large expert gets wrong too, paying
for nothing; `large-helps` (small F1 < tau and large F1 >= tau) escalates only when it pays off.
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from groupproject.evidence import tokenize
from groupproject.experts import AUX_VERBS

log = logging.getLogger(__name__)

CUE_COLUMNS = ["q_starts_aux", "q_has_or", "q_comparative", "q_num_tokens", "q_num_capitalised"]
EVIDENCE_COLUMNS = ["ev_top1", "ev_top2_sum", "ev_margin_2_3", "ev_entropy", "ev_sufficiency"]
ROUTER_INPUTS = {"question": CUE_COLUMNS, "question+evidence": CUE_COLUMNS + EVIDENCE_COLUMNS}
COMPARATIVE = frozenset("both same first older younger earlier later more larger longer higher".split())


def add_question_cues(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    toks = df["question"].map(tokenize)
    df["q_starts_aux"] = toks.map(lambda t: float(bool(t) and t[0] in AUX_VERBS))
    df["q_has_or"] = toks.map(lambda t: float("or" in t))
    df["q_comparative"] = toks.map(lambda t: float(bool(COMPARATIVE & set(t))))
    df["q_num_tokens"] = toks.map(len).astype(float)
    df["q_num_capitalised"] = df["question"].str.count(r"\b[A-Z]").astype(float)
    return df


LABELS = ("small-fails", "large-helps")


def escalation_label(df: pd.DataFrame, tau: float, label: str) -> np.ndarray:
    """1 if the question should be escalated under `label` (see module docstring)."""
    small_wrong = df["small_f1"] < tau
    if label == "small-fails":
        return small_wrong.astype(int).to_numpy()
    if label == "large-helps":
        return (small_wrong & (df["large_f1"] >= tau)).astype(int).to_numpy()
    raise ValueError(f"unknown label: {label}")


def usable_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Drop columns that are entirely missing (e.g. sufficiency with the lexical scorer)."""
    return [c for c in columns if df[c].notna().any()]


def build_router(dense_columns: list[str]) -> Pipeline:
    features = ColumnTransformer([
        ("text", HashingVectorizer(ngram_range=(1, 2), n_features=2**18, alternate_sign=False, norm="l2"), "question"),
        ("dense", StandardScaler(), dense_columns),
    ])
    return Pipeline([("features", features), ("clf", LogisticRegression(C=1.0, max_iter=2000))])


def predict_escalation(router: dict, df: pd.DataFrame) -> np.ndarray:
    """P(small expert fails) for each row."""
    X = add_question_cues(df)
    X[router["columns"]] = X[router["columns"]].fillna(0.0)
    return router["model"].predict_proba(X)[:, 1]


def train_routers(df: pd.DataFrame, tau: float) -> dict[str, dict]:
    """One router per (input set, label), named "<inputs>/<label>"."""
    X = add_question_cues(df)
    routers = {}
    for label in LABELS:
        y = escalation_label(df, tau, label)
        if len(set(y)) < 2:
            log.warning(f"  skipping label {label}: all {len(y)} training labels are {y[0]}")
            continue
        for inputs, columns in ROUTER_INPUTS.items():
            cols = usable_columns(X, columns)
            X[cols] = X[cols].fillna(0.0)
            model = build_router(cols).fit(X, y)
            routers[f"{inputs}/{label}"] = {"model": model, "columns": cols, "label": label}
            log.info(f"  {inputs}/{label}: {len(cols)} dense features {cols}")
    if not routers:
        raise ValueError(f"no label has both classes on {len(df)} questions; change tau or add questions")
    return routers


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=0.8, help="small expert counts as correct at F1 >= tau")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = pd.read_parquet(args.run / "router_train.parquet")
    rates = " | ".join(f"{label} {escalation_label(df, args.tau, label).mean():.3f}" for label in LABELS)
    log.info(f"router_train: {len(df)} questions, escalate rate at tau={args.tau}: {rates}")
    routers = train_routers(df, args.tau)
    out = args.run / "routers.pkl"
    out.write_bytes(pickle.dumps({"tau": args.tau, "routers": routers}))
    log.info(f"Saved {out}")


if __name__ == "__main__":
    main()
