"""Download and prepare the HotpotQA dataset for experiments."""

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

CONFIG = "distractor"
OUTPUT_DIR = Path("./data/hotpotqa")
SEED = 42
SAMPLE = None  # Set to e.g. 500 for quick iteration


def extract_supporting_context(example: dict) -> str:
    """Return the concatenated sentences marked as supporting facts."""
    title_to_sents = {}
    for title, sentences in zip(example["context"]["title"], example["context"]["sentences"]):
        title_to_sents[title] = sentences

    parts = []
    for title, sent_id in zip(example["supporting_facts"]["title"], example["supporting_facts"]["sent_id"]):
        sents = title_to_sents.get(title, [])
        if 0 <= sent_id < len(sents):
            parts.append(sents[sent_id].strip())

    return " ".join(parts)


def build_full_context(example: dict) -> str:
    """Concatenate all context paragraphs into a single string with titles as headers."""
    parts = []
    for title, sentences in zip(example["context"]["title"], example["context"]["sentences"]):
        para = " ".join(s.strip() for s in sentences)
        parts.append(f"[{title}] {para}")
    return "\n\n".join(parts)


def flatten_example(example: dict) -> dict:
    """Flatten a raw HotpotQA example into a dict suitable for a DataFrame."""
    return {
        "id": example["id"],
        "question": example["question"],
        "answer": example["answer"],
        "type": example["type"],
        "level": example["level"],
        "supporting_context": extract_supporting_context(example),
        "full_context": build_full_context(example),
        "num_context_paragraphs": len(example["context"]["title"]),
        "num_supporting_facts": len(example["supporting_facts"]["title"]),
    }


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Missing dependency: pip install datasets pyarrow")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading HotpotQA ({CONFIG}) …")
    dataset = load_dataset("hotpotqa/hotpot_qa", name=CONFIG)

    for split_name, split_data in dataset.items():
        if SAMPLE is not None and SAMPLE < len(split_data):
            split_data = split_data.shuffle(seed=SEED).select(range(SAMPLE))

        records = [flatten_example(ex) for ex in split_data]
        df = pd.DataFrame(records)
        log.info(f"{split_name}: {len(df)} examples | types={df['type'].value_counts().to_dict()} | levels={df['level'].value_counts().to_dict()}")

        base = OUTPUT_DIR / f"{CONFIG}_{split_name}"
        df.to_json(base.with_suffix(".json"), orient="records", lines=True, force_ascii=False)
        df.to_parquet(base.with_suffix(".parquet"), index=False, engine="pyarrow")
        log.info(f"  Saved {base}.{{json,parquet}}")

    log.info("Done.")


if __name__ == "__main__":
    main()
