"""Experiment E1's pretraining arms: nano-jev v1.0 retrained from its own recipe, with and without HotpotQA.

The recipe is nano-jev's public code (github.com/shubham10divakar/nano-jev, Apache-2.0), cloned at a pinned commit
and run unchanged: its data builders (`nanojev.data`), `scripts/train.py` and `scripts/evaluate.py`. v1.0's card
fixes the settings: base microsoft/MiniLM-L12-H384-uncased, 3 epochs, lr 5e-5, batch 16, seed 0, best epoch by dev NLL.

    repro     v1.0's own committed data (data/{train,calib,test}.jsonl: 6,000 HotpotQA + 6,000 SQuAD 2.0 + 8,000
              MultiNLI training questions): a retrain, so the gap to v1.0 measures retraining noise
    nohotpot  the same builders with HotpotQA replaced by as many SQuAD 2.0 questions (12,000 SQuAD + 8,000 MultiNLI),
              and calib / test without HotpotQA, so not even epoch selection sees it

    nanojev-pretrain fetch --out runs/nanojev-pretrain          # login node: clone + dataset downloads, no compute
    nanojev-pretrain train --variant repro --out runs/nanojev-pretrain   # GPU: build (nohotpot), train, evaluate

Each variant's model lands in `<out>/<variant>/model` (a judge `init`); `<out>/<variant>/eval-*.json` are nano-jev's
own reports on its calib/test split and on v1.0's test split, beside v1.0's on the same split (`<out>/v1.0/`): the
gate before the variant is used.
"""

import argparse
import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path

from stair import provenance
from stair.evidence import resolve_checkpoint

log = logging.getLogger(__name__)

REPO = "https://github.com/shubham10divakar/nano-jev.git"
COMMIT = "25b4655ac129c7151eb39d2ed1d15227f93f1ba6"
BASE = "microsoft/MiniLM-L12-H384-uncased"
V1 = "sdmlai/nano-jev@v1.0"
HPARAMS = ["--epochs", "3", "--lr", "5e-5", "--batch-size", "16", "--max-length", "512"]  # v1.0's card
DATASETS = [("rajpurkar/squad_v2", None), ("nyu-mll/multi_nli", None), ("hotpotqa/hotpot_qa", "distractor")]
NOHOTPOT = {"squad_train": 12000, "mnli_train": 8000, "squad_eval": 500, "mnli_eval": 500}
VARIANTS = ("repro", "nohotpot")


def src_dir(out: Path) -> Path:
    return out / "nano-jev-src"


def fetch(out: Path) -> None:
    """Clone the recipe at COMMIT and download its datasets into the HF cache (login node: downloads only)."""
    src = src_dir(out)
    if not src.exists():
        subprocess.run(["git", "clone", "-q", REPO, str(src)], check=True)
    subprocess.run(["git", "-C", str(src), "checkout", "-q", COMMIT], check=True)
    from datasets import load_dataset

    for name, config in DATASETS:
        load_dataset(name, config)
        log.info(f"cached {name}")
    for model in (BASE, V1):
        log.info(f"cached {model} at {resolve_checkpoint(model)}")
    log.info(f"nano-jev recipe at {COMMIT[:7]} in {src}")


def build_nohotpot(src: Path, seed: int = 0) -> dict[str, list[dict]]:
    """nanojev.data.build_all with HotpotQA left out of train, calib and test (its builders, its rng order)."""
    sys.path.insert(0, str(src))
    from nanojev.data import mnli_examples, squad2_examples

    rng = random.Random(seed)
    train = squad2_examples("train", NOHOTPOT["squad_train"], rng) + mnli_examples("train", NOHOTPOT["mnli_train"], rng)
    calib, test = [], []
    for part in (squad2_examples("validation", 2 * NOHOTPOT["squad_eval"], rng),
                 mnli_examples("validation_matched", 2 * NOHOTPOT["mnli_eval"], rng)):
        rng.shuffle(part)
        calib += part[: len(part) // 2]
        test += part[len(part) // 2:]
    rng.shuffle(train)
    return {"train": train, "calib": calib, "test": test}


def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def source_counts(data: Path) -> dict[str, dict[str, int]]:
    out = {}
    for split in ("train", "calib", "test"):
        counts: dict[str, int] = {}
        with open(data / f"{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                counts[f"{ex['source']}/{ex['decision']}"] = counts.get(f"{ex['source']}/{ex['decision']}", 0) + 1
        out[split] = counts
    return out


def run(cmd: list[str], src: Path) -> None:
    log.info(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=src)


def evaluate(src: Path, model: Path, data: Path, test_data: Path, results: Path, name: str, save: bool) -> None:
    cmd = [sys.executable, "scripts/evaluate.py", "--model", str(model), "--data", str(data),
           "--test-data", str(test_data), "--results-dir", str(results), "--name", name]
    run(cmd + ([] if save else ["--no-save"]), src)


def train(out: Path, variant: str, seed: int) -> None:
    src = src_dir(out).resolve()
    if not (src / "scripts" / "train.py").exists():
        raise SystemExit(f"{src}: run `nanojev-pretrain fetch` on the login node first")
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    if head.stdout.strip() != COMMIT:
        raise SystemExit(f"{src} is at {head.stdout.strip()[:7]}, not the pinned {COMMIT[:7]}")
    target = (out / variant).resolve()
    model = target / "model"
    if variant == "repro":
        data = src / "data"  # v1.0's committed training data
    else:
        data = target / "data"
        if not (data / "test.jsonl").exists():
            for split, examples in build_nohotpot(src, seed).items():
                write_jsonl(examples, data / f"{split}.jsonl")
    (target / "data-sources.json").parent.mkdir(parents=True, exist_ok=True)
    (target / "data-sources.json").write_text(json.dumps(source_counts(data), indent=1))
    if not (model / "nanojev_config.json").exists():
        run([sys.executable, "scripts/train.py", "--data", str(data), "--out", str(model), "--base", BASE,
             "--seed", str(seed), *HPARAMS], src)
    evaluate(src, model, data, data, target, "eval-own", save=True)  # fits calibration.json on its own calib
    evaluate(src, model, data, src / "data", target, "eval-v1-test", save=False)
    v1 = out.resolve() / "v1.0"
    if not (v1 / "eval-v1-test.json").exists():  # the reference: released v1.0 on the same split
        evaluate(src, resolve_checkpoint(V1), src / "data", src / "data", v1, "eval-v1-test", save=False)
    provenance.record(target, "nanojev-pretrain", variant=variant, seed=seed, commit=COMMIT, base=BASE,
                      base_revision=provenance.hub_revision(BASE), **provenance.environment())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["fetch", "train"])
    parser.add_argument("--out", type=Path, default=Path("runs/nanojev-pretrain"))
    parser.add_argument("--variant", choices=VARIANTS, action="append", help="train: repeatable (default both)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.step == "fetch":
        fetch(args.out)
        return
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")  # compute nodes: everything comes from `fetch`'s cache
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    for variant in args.variant or VARIANTS:
        train(args.out, variant, args.seed)


if __name__ == "__main__":
    main()
