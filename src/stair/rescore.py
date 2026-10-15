"""Experiment 8: re-score the cached long-form answers with an LLM grader and short-answer extraction.

The lenient scorer (gold answer contained anywhere in the response) rewards long answers. Pasting
nano-jev's top 5 passages as the "answer" scores 0.798 lenient, above every model. This module
re-scores every cached answer in `<out>/answers/<split>/<cell>.parquet` two ways, with one fixed,
ungated grader model (default Qwen2.5-14B-Instruct):

- grade: the grader reads (question, gold answer, response) and must say CORRECT or INCORRECT. The
  verdict is the probability of the two words, from the exact log-likelihood of each as the
  assistant's reply (no sampling, so it is deterministic). A response counts as correct only if it
  commits to the gold answer. Hedging, lists of candidates and a different first answer count as
  incorrect.
- extract: the grader states the response's final answer in a few words (greedy), and the official
  HotpotQA EM / F1 are computed on it (`stair.metrics`). That keeps experiment 8 comparable with
  experiments 1-7.

    rescore validate --grader Qwen/Qwen2.5-14B-Instruct     # must pass before anything else
    rescore run      --split test                           # every cell of that split
    rescore summary

`validate` checks the grader on two controls and writes `<out>/rescore/validation.json`:

- Known short answers, from experiment 1's 7B answers to the same 500 test questions with official
  EM: it should grade EM = 1 as CORRECT and F1 = 0 as INCORRECT.
- A negative control: nano-jev's top 5 passages pasted as the response. It should grade almost all
  of them INCORRECT. This is the failure the lenient scorer has.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stair import provenance
from stair.experts import clean_answer
from stair.metrics import exact_match, f1_score

log = logging.getLogger(__name__)

GRADER = "Qwen/Qwen2.5-14B-Instruct"
GRADE_SYSTEM = "You are a strict, careful grader of answers to factual questions."
GRADE_USER = """Question: {question}
Gold answer: {gold}
Response: {response}

Does the response answer the question correctly?
- CORRECT: the response commits to an answer that means the same as the gold answer. Different wording, spelling, formatting, units or date order is fine, and so is extra explanation, as long as the answer it commits to matches the gold answer.
- INCORRECT: the response commits to a different answer; says it cannot answer or that the information is missing; hedges between several candidate answers; or mentions the gold answer only in passing (for example while quoting references) but concludes something else.
Reply with exactly one word: CORRECT or INCORRECT."""
EXTRACT_SYSTEM = "You extract the final answer from a response, in as few words as possible."
EXTRACT_USER = """Question: {question}
Response: {response}

What final answer does the response give to the question? Reply with only that answer, in as few words as possible: a name, date, number, place or short phrase, or "yes" / "no" for yes/no questions. Do not add any explanation. If the response does not commit to one answer, reply "none"."""
VERDICTS = ("CORRECT", "INCORRECT")

# The grader passes validation only if (validation thresholds, fixed before any run):
MIN_AGREE_EM1 = 0.90     # share of EM = 1 short answers graded CORRECT
MIN_AGREE_F10 = 0.90     # share of F1 = 0 short answers graded INCORRECT
MAX_DUMP_CORRECT = 0.10  # share of pasted-passage "responses" graded CORRECT


class Grader:
    """A chat LLM that scores fixed continuations and extracts short answers."""

    def __init__(self, model: str, device: str | None = None, revision: str | None = None):
        from stair.pool_answer import Generator

        self.gen = Generator(model, device, revision=revision)
        self.tok, self.model, self.torch = self.gen.tokenizer, self.gen.model, self.gen.torch
        self.verdict_ids = [self.tok(v, add_special_tokens=False)["input_ids"] for v in VERDICTS]

    def verdict_logprobs(self, prompts: list[str], batch: int) -> np.ndarray:
        """[n, 2] summed log-probabilities of 'CORRECT' and 'INCORRECT' as the assistant's reply.

        Prompts are batched in length order, so a batch holds similar lengths. In bfloat16, heavy left
        padding moved P(CORRECT) by up to 0.077 for the same answer (Qwen2.5-0.5B; float32 agreed to
        2e-5), so `batch = 1`, the default of `rescore`, makes every grade independent of the others.
        """
        torch = self.torch
        out = np.zeros((len(prompts), len(VERDICTS)))
        order = np.argsort([len(p) for p in prompts], kind="stable")
        prompts = [prompts[i] for i in order]
        prefix = [self.tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        pad = self.tok.pad_token_id
        for v, cont in enumerate(self.verdict_ids):
            for i in range(0, len(prompts), batch):
                seqs = [p + cont for p in prefix[i: i + batch]]
                width = max(map(len, seqs))
                ids = torch.full((len(seqs), width), pad, dtype=torch.long)
                mask = torch.zeros((len(seqs), width), dtype=torch.long)
                for r, s in enumerate(seqs):  # left padding: every continuation ends at the last position
                    ids[r, width - len(s):] = torch.tensor(s)
                    mask[r, width - len(s):] = 1
                # A plain forward pass numbers positions from the first (pad) token; left-padded rows need
                # positions counted from their first real token, as generate() does, or their scores shift.
                positions = (mask.cumsum(-1) - 1).clamp(min=0)
                dev = self.gen.input_device
                with torch.no_grad():
                    logits = self.model(input_ids=ids.to(dev), attention_mask=mask.to(dev), position_ids=positions.to(dev),
                                        logits_to_keep=len(cont) + 1).logits[:, :-1].float()
                logp = torch.log_softmax(logits, dim=-1)
                target = torch.tensor(cont, device=logp.device).expand(len(seqs), -1)
                out[i: i + len(seqs), v] = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1).sum(-1).cpu().numpy()
        restored = np.zeros_like(out)
        restored[order] = out  # back to the caller's order
        return restored

    def grade(self, questions, golds, responses, batch: int = 8) -> np.ndarray:
        """P(CORRECT) per response, from the two verdicts' log-likelihoods."""
        prompts = [self.gen.chat(GRADE_SYSTEM, GRADE_USER.format(question=q, gold=g, response=r))
                   for q, g, r in zip(questions, golds, responses)]
        lp = self.verdict_logprobs(prompts, batch)
        lp -= lp.max(axis=1, keepdims=True)
        p = np.exp(lp)
        return p[:, 0] / p.sum(axis=1)

    def extract(self, questions, responses, batch: int = 16, max_new_tokens: int = 24) -> list[str]:
        prompts = [self.gen.chat(EXTRACT_SYSTEM, EXTRACT_USER.format(question=q, response=r))
                   for q, r in zip(questions, responses)]
        order = np.argsort([len(p) for p in prompts], kind="stable")  # similar lengths per batch: little padding
        out = [""] * len(prompts)
        for i in range(0, len(order), batch):
            idx = order[i: i + batch]
            res = self.gen.generate([prompts[j] for j in idx], 1, 0.0, max_new_tokens, seed=0)
            for j, r in zip(idx, res):
                out[j] = clean_answer(r["texts"][0]).strip('"').strip("'")
        return out


def score_frame(grader: Grader, questions, golds, responses, batch: int, extract_batch: int = 16) -> pd.DataFrame:
    p = grader.grade(questions, golds, responses, batch)
    ext = grader.extract(questions, responses, extract_batch)
    none = [e.strip().lower() in ("none", "") for e in ext]
    return pd.DataFrame({
        "p_correct": p, "correct": (p >= 0.5).astype(float), "extracted": ext,
        "ext_em": [0.0 if n else exact_match(e, g) for e, g, n in zip(ext, golds, none)],
        "ext_f1": [0.0 if n else f1_score(e, g) for e, g, n in zip(ext, golds, none)],
    })


def questions_of(out: Path, split: str) -> pd.DataFrame:
    return pd.read_parquet(out / "pools" / f"{split}.parquet", columns=["id", "question", "answer"]).set_index("id")


# ---------------------------------------------------------------------------------------------


def cmd_validate(args) -> None:
    from stair.pool_answer import top_passages

    out = args.out
    q = questions_of(out, "test")
    grader = Grader(args.grader, args.device, args.revision)
    provenance.record(out / "rescore", "rescore-validate", grader=args.grader,
                      revision=provenance.hub_revision(args.grader, args.revision), **provenance.environment())
    # 1. Known short answers: experiment 1's 7B (all 10 passages) on the same questions, with official EM / F1.
    known = pd.read_parquet(args.known, columns=["id", "large_pred", "large_em", "large_f1"]).set_index("id")
    known = known.loc[[i for i in q.index if i in known.index]]
    start = time.perf_counter()
    pk = grader.grade(q.loc[known.index, "question"].tolist(), q.loc[known.index, "answer"].tolist(),
                      known["large_pred"].tolist(), args.batch)
    em1, f10 = known["large_em"].to_numpy() == 1, known["large_f1"].to_numpy() == 0
    # 2. Negative control: nano-jev's top 5 passages pasted as the response.
    pools = pd.read_parquet(out / "pools" / "test.parquet").merge(
        pd.read_parquet(out / "scores" / "test.parquet", columns=["id", "nanojev"]), on="id").set_index("id")
    dumps = [" ".join(top_passages(r, r["nanojev"], 5)) for _, r in pools.loc[q.index].iterrows()]
    pd_ = grader.grade(q["question"].tolist(), q["answer"].tolist(), dumps, args.batch)
    seconds = time.perf_counter() - start
    report = {
        "grader": args.grader, "revision": provenance.hub_revision(args.grader, args.revision),
        "known_n": int(len(known)), "known_em1_n": int(em1.sum()), "known_f10_n": int(f10.sum()),
        "agree_em1_correct": float((pk[em1] >= 0.5).mean()), "agree_f10_incorrect": float((pk[f10] < 0.5).mean()),
        "known_graded_correct": float((pk >= 0.5).mean()), "known_em": float(em1.mean()),
        "known_f1": float(known["large_f1"].mean()),
        "dump_n": len(dumps), "dump_graded_correct": float((pd_ >= 0.5).mean()),
        "thresholds": {"min_agree_em1": MIN_AGREE_EM1, "min_agree_f10": MIN_AGREE_F10, "max_dump_correct": MAX_DUMP_CORRECT},
        "seconds": seconds,
    }
    report["passed"] = bool(report["agree_em1_correct"] >= MIN_AGREE_EM1 and report["agree_f10_incorrect"] >= MIN_AGREE_F10
                            and report["dump_graded_correct"] <= MAX_DUMP_CORRECT)
    # Disagreements for a human (or Claude) to read: EM = 1 graded INCORRECT, F1 = 0 graded CORRECT.
    dis = known.assign(p_correct=pk, question=q.loc[known.index, "question"], gold=q.loc[known.index, "answer"])
    dis = dis[(em1 & (pk < 0.5)) | (f10 & (pk >= 0.5))]
    dis[["question", "gold", "large_pred", "large_em", "large_f1", "p_correct"]].head(40).to_csv(
        out / "rescore" / "validation-disagreements.csv")
    (out / "rescore" / "validation.json").write_text(json.dumps(report, indent=1))
    log.info(json.dumps(report, indent=1))
    log.info("VALIDATION " + ("PASSED" if report["passed"] else "FAILED") + f" (see {out / 'rescore' / 'validation.json'})")


def cmd_run(args) -> None:
    out = args.out
    if not args.skip_validation_check:
        v = out / "rescore" / "validation.json"
        if not v.exists() or not json.loads(v.read_text()).get("passed"):
            raise SystemExit(f"{v} is missing or did not pass: run `rescore validate` first (or --skip-validation-check)")
    q = questions_of(out, args.split)
    cells = sorted((out / "answers" / args.split).glob("*.parquet"))
    if args.only:
        cells = [c for c in cells if c.stem in args.only]
    grader = None
    for f in cells:
        target = out / "rescore" / args.split / f.name
        if target.exists():
            log.info(f"{args.split}/{f.stem}: done")
            continue
        grader = grader or Grader(args.grader, args.device, args.revision)
        provenance.record(out / "rescore", "rescore-run", split=args.split, cell=f.stem, grader=args.grader,
                          revision=provenance.hub_revision(args.grader, args.revision), **provenance.environment())
        a = pd.read_parquet(f, columns=["id", "pred", "lenient"])
        parts = out / "rescore" / args.split / f"{f.stem}.parts"
        parts.mkdir(parents=True, exist_ok=True)
        chunks = [a.iloc[i: i + args.chunk_size] for i in range(0, len(a), args.chunk_size)]
        start = time.perf_counter()
        for c, chunk in enumerate(chunks):
            pf = parts / f"{c:05d}.parquet"
            if pf.exists():
                continue
            s = score_frame(grader, q.loc[chunk["id"], "question"].tolist(), q.loc[chunk["id"], "answer"].tolist(),
                            chunk["pred"].tolist(), args.batch, args.extract_batch)
            s.insert(0, "id", chunk["id"].to_numpy())
            s["lenient"] = chunk["lenient"].to_numpy()
            s.to_parquet(pf)
        merged = pd.concat([pd.read_parquet(p) for p in sorted(parts.glob("*.parquet"))], ignore_index=True)
        merged.to_parquet(target)
        log.info(f"{args.split}/{f.stem}: {len(merged)} answers in {time.perf_counter() - start:.0f} s | "
                 f"lenient {merged['lenient'].mean():.3f} graded-correct {merged['correct'].mean():.3f} "
                 f"extracted EM {merged['ext_em'].mean():.3f} F1 {merged['ext_f1'].mean():.3f}")


def cmd_summary(args) -> None:
    rows = []
    for split_dir in sorted((args.out / "rescore").iterdir()):
        if not split_dir.is_dir():
            continue
        for f in sorted(split_dir.glob("*.parquet")):
            d = pd.read_parquet(f)
            units = pd.read_parquet(args.out / "answers" / split_dir.name / f.name, columns=["units"])["units"].mean()
            rows.append({"split": split_dir.name, "cell": f.stem, "n": len(d), "lenient": d["lenient"].mean(),
                         "graded_correct": d["correct"].mean(), "ext_em": d["ext_em"].mean(), "ext_f1": d["ext_f1"].mean(),
                         "units": units})
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "rescore" / "summary.csv", index=False)
    with pd.option_context("display.width", 200):
        log.info("\n" + table.round(3).to_string(index=False))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["validate", "run", "summary"])
    parser.add_argument("--out", type=Path, default=Path("runs/exp8"))
    parser.add_argument("--grader", default=GRADER)
    parser.add_argument("--revision", help="grader revision to pin (default: whatever the cache holds)")
    parser.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--split", default="test", help="run: answers/<split>/")
    parser.add_argument("--only", action="append", default=[], help="run: cell names (default all); repeatable")
    parser.add_argument("--batch", type=int, default=1,
                        help="responses per grading forward pass (1 = exact; larger batches shift bf16 grades slightly)")
    parser.add_argument("--extract-batch", type=int, default=16, help="run: responses per extraction batch (length-sorted)")
    parser.add_argument("--chunk-size", type=int, default=250, help="run: answers per resumable chunk")
    parser.add_argument("--known", type=Path, default=Path("results/gpu/test.parquet"),
                        help="validate: experiment 1's cache with short answers and official EM/F1")
    parser.add_argument("--skip-validation-check", action="store_true", help="run: do not require a passed validation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    (args.out / "rescore").mkdir(parents=True, exist_ok=True)
    {"validate": cmd_validate, "run": cmd_run, "summary": cmd_summary}[args.step](args)


if __name__ == "__main__":
    main()
