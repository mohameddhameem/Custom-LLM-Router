"""Experiment 8: re-score the cached long-form answers with an LLM grader and short-answer extraction.

The lenient scorer (gold answer contained anywhere in the response) rewards long answers. Pasting
nano-jev's top 5 passages as the "answer" scores 0.798 lenient, above every model. This module
re-scores every cached answer in `<out>/answers/<split>/<cell>.parquet` two ways, with one fixed,
ungated grader model (default Qwen2.5-14B-Instruct):

Version 3 extracts first, then grades:

1. extract: the grader states the response's committed final answer in a few words (greedy), or
   "none" when the response commits to no single answer. The official HotpotQA EM / F1 are computed
   on it (`stair.metrics`), which keeps experiment 8 comparable with experiments 1-7.
2. grade the extracted answer (primary metric, `correct`; "none" is wrong). The grader reads
   (question, gold answer, short answer) and gives a one-word verdict, CORRECT or INCORRECT. P(CORRECT)
   is the next-token mass of every vocabulary token that begins "CORRECT" against every token that
   begins "INCORRECT", case-insensitive, so it works for any tokenisation. The gold answer is correct by
   definition.

Also recorded:
- `correct_first`: the grade of the response's first sentence, a length-controlled check;
- `correct_full`: the grade of the whole response. This is version 2's metric and a diagnostic only,
  because it accepted about 83% of pasted-passage dumps that contain the gold.

    rescore validate --grader Qwen/Qwen2.5-14B-Instruct     # must pass before anything else
    rescore run      --split test                           # every cell of that split
    rescore summary

`validate` writes `<out>/rescore/validation.json`. Gating uses a slice not used while the method was
designed. The checks run the full extract-then-grade pipeline:

- Known short answers: experiment 1's 7B answers to HotpotQA dev items 1000-1499, with official EM /
  F1. Extraction must leave a short answer unchanged (faithfulness). EM = 1 should be graded CORRECT.
  F1 = 0 should mostly be INCORRECT; HotpotQA's gold strings have aliases ("AdScam" for "The
  sponsorship scandal"), so a good grader cannot reach 100% there.
- A negative control: nano-jev's top 5 passages pasted as the "response", on the calib pools. They
  commit to no answer, so extraction should give "none" or a wrong answer for almost all of them.
- A self-check: the grader's greedy, written verdict must agree with its scored P(CORRECT), and almost
  all next-token mass must fall on the two verdicts. A disagreement means a computation or format bug.

History (2026-10-15):
- Version 1 failed: the grader graded against its own knowledge (`results/exp8-rescore/README.md`).
- Version 2 fixed that, with 0.996 / 0.917 on short answers. It still graded 0.619 of pasted-passage
  dumps CORRECT, because the 14B treats "the gold is stated somewhere" as answering
  (`results/exp8-rescore/v2/README.md`).
- Version 3 therefore grades only the extracted short answer, where version 2 was reliable.

The slice used for diagnosis (dev items 500-999, experiment 8's test questions) is still reported, as a
diagnostic only.
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stair import provenance
from stair.experts import clean_answer
from stair.metrics import exact_match, f1_score

log = logging.getLogger(__name__)

GRADER = "Qwen/Qwen2.5-14B-Instruct"
GRADE_SYSTEM = ("You compare a response with a reference answer. The reference (gold) answer is correct by definition: "
                "never use your own knowledge to second-guess it, and do not judge whether the response is true in general.")
GRADE_USER = """Question: {question}
Gold answer: {gold}
Response: {response}

Does the response give the same final answer as the gold answer?
- CORRECT: the response commits to one final answer, and it matches the gold answer: the same entity, date, number, place, or yes/no. Different wording, spelling, aliases, abbreviations, formatting, units or date order are fine, and so is extra explanation.
- INCORRECT: the response's final answer differs from the gold answer; or it says it cannot answer or that the information is missing; or it hedges between several candidates; or it gives no final answer at all (for example it only quotes, lists or summarises references or passages); or the gold answer appears only inside quoted material while the response concludes something else.
Reply with exactly one word: CORRECT or INCORRECT."""
EXTRACT_SYSTEM = ("You report the final answer that a response commits to, in as few words as possible. "
                  "You never answer the question yourself.")
EXTRACT_USER = """Question: {question}
Response: {response}

What final answer does the response itself commit to? Rules:
- Report only an answer the response explicitly states as its answer to the question (a direct answer, or a concluding statement such as "so the answer is X").
- Do not answer the question yourself, and do not work out an answer from facts, passages or references the response quotes, lists or summarises without concluding.
- If the response hedges between candidates, says it cannot answer, or gives no conclusion, reply "none".
Reply with only that answer, in as few words as possible: a name, date, number, place or short phrase, or "yes" / "no" for yes/no questions. Do not add any explanation."""
VERDICTS = ("CORRECT", "INCORRECT")

# Version 3 validation thresholds, fixed on 2026-10-15 before any version 3 run:
MIN_AGREE_EM1 = 0.95         # EM = 1 short answers, extracted then graded CORRECT
MIN_AGREE_F10 = 0.80         # F1 = 0 short answers graded INCORRECT (gold aliases make 100% unreachable)
MAX_DUMP_CORRECT = 0.10      # pasted-passage dumps extracted then graded CORRECT. An extractor may fairly read a
                             # dump that opens with a direct statement of the answer; 0.10 is half of the 0.209 of
                             # calib dumps whose first sentence contains the gold
MIN_GEN_AGREEMENT = 0.95     # the written greedy verdict agrees with P(CORRECT) >= 0.5
MIN_VERDICT_MASS = 0.90      # mean next-token mass on the two verdicts (format adherence)
MIN_EXTRACT_FAITHFUL = 0.90  # extraction leaves a short answer unchanged (official EM between the two)
NONE_ANSWERS = {"none", ""}


def first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]


def is_none(extracted: str) -> bool:
    return extracted.strip().strip(".").lower() in NONE_ANSWERS


def verdict_token_sets(tokenizer) -> tuple[np.ndarray, np.ndarray]:
    """Vocabulary ids whose text (stripped, upper-cased) is a non-empty prefix of CORRECT / of INCORRECT."""
    correct, incorrect = [], []
    for i in range(len(tokenizer)):
        text = tokenizer.decode([i]).strip().upper()
        if not text:
            continue
        if VERDICTS[0].startswith(text):
            correct.append(i)
        elif VERDICTS[1].startswith(text):
            incorrect.append(i)
    return np.asarray(correct, dtype=np.int64), np.asarray(incorrect, dtype=np.int64)


class Grader:
    """A chat LLM that grades responses from its next-token verdict mass, and extracts short answers."""

    def __init__(self, model: str, device: str | None = None, revision: str | None = None, dtype: str = "bfloat16"):
        from stair.pool_answer import Generator

        self.gen = Generator(model, device, dtype=dtype, revision=revision)
        self.tok, self.model, self.torch = self.gen.tokenizer, self.gen.model, self.gen.torch
        self.correct_ids, self.incorrect_ids = verdict_token_sets(self.tok)

    def verdict_probs(self, prompts: list[str], batch: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """P(CORRECT) and the verdict mass (P(CORRECT-prefix) + P(INCORRECT-prefix)) for each prompt.

        Prompts run in length order, so a batch holds similar lengths. One prompt per pass (`batch = 1`,
        the default) makes each grade independent of the others: with heavy left padding in bfloat16
        the same answer's probability moved by up to 0.077 (Qwen2.5-0.5B; float32 agreed to 2e-5).
        """
        torch = self.torch
        p_correct = np.full(len(prompts), 0.5)
        mass = np.zeros(len(prompts))
        if not (len(self.correct_ids) and len(self.incorrect_ids)):
            log.warning("the tokenizer has no CORRECT / INCORRECT prefix tokens: every grade is 0.5")
            return p_correct, mass
        order = np.argsort([len(p) for p in prompts], kind="stable")
        dev = self.gen.input_device
        c_ids, i_ids = torch.tensor(self.correct_ids, device=dev), torch.tensor(self.incorrect_ids, device=dev)
        pad = self.tok.pad_token_id
        for s in range(0, len(order), batch):
            idx = order[s: s + batch]
            seqs = [self.tok(prompts[j], add_special_tokens=False)["input_ids"] for j in idx]
            width = max(map(len, seqs))
            ids = torch.full((len(seqs), width), pad, dtype=torch.long)
            mask = torch.zeros((len(seqs), width), dtype=torch.long)
            for r, q in enumerate(seqs):  # left padding: every prompt ends at the last position
                ids[r, width - len(q):] = torch.tensor(q)
                mask[r, width - len(q):] = 1
            positions = (mask.cumsum(-1) - 1).clamp(min=0)  # count from each row's first real token, as generate() does
            with torch.no_grad():
                logits = self.model(input_ids=ids.to(dev), attention_mask=mask.to(dev), position_ids=positions.to(dev),
                                    logits_to_keep=1).logits
            last = torch.softmax(logits[:, -1, :].float(), dim=-1)  # [:, -1] is the last position whether or not
            pc, pi = last[:, c_ids].sum(-1), last[:, i_ids].sum(-1)  # logits_to_keep was honoured
            p_correct[idx] = (pc / (pc + pi).clamp(min=1e-12)).cpu().numpy()
            mass[idx] = (pc + pi).cpu().numpy()
        return p_correct, mass

    def grade_prompts(self, questions, golds, responses) -> list[str]:
        return [self.gen.chat(GRADE_SYSTEM, GRADE_USER.format(question=q, gold=g, response=r))
                for q, g, r in zip(questions, golds, responses)]

    def grade(self, questions, golds, responses, batch: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """(P(CORRECT), verdict mass) per response."""
        return self.verdict_probs(self.grade_prompts(questions, golds, responses), batch)

    def written_verdicts(self, questions, golds, responses, batch: int = 8) -> np.ndarray:
        """The greedy written verdict: 1.0 CORRECT, 0.0 INCORRECT, nan when it wrote neither (a self-check)."""
        prompts = self.grade_prompts(questions, golds, responses)
        out = np.full(len(prompts), np.nan)
        order = np.argsort([len(p) for p in prompts], kind="stable")
        for s in range(0, len(order), batch):
            idx = order[s: s + batch]
            res = self.gen.generate([prompts[j] for j in idx], 1, 0.0, 4, seed=0)
            for j, r in zip(idx, res):
                text = r["texts"][0].strip().upper()
                out[j] = 0.0 if text.startswith(VERDICTS[1]) else 1.0 if text.startswith(VERDICTS[0]) else np.nan
        return out

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


def extract_then_grade(grader: Grader, questions, golds, responses, batch: int, extract_batch: int = 16):
    """(extracted answers, none flags, P(CORRECT) of the extracted answer with 'none' = 0, verdict mass)."""
    questions, golds = list(questions), list(golds)
    ext = grader.extract(questions, list(responses), extract_batch)
    none = np.array([is_none(e) for e in ext])
    p, mass = grader.grade(questions, golds, ["none" if n else e for e, n in zip(ext, none)], batch)
    return ext, none, np.where(none, 0.0, p), mass


def score_frame(grader: Grader, questions, golds, responses, batch: int, extract_batch: int = 16) -> pd.DataFrame:
    questions, golds, responses = list(questions), list(golds), list(responses)
    ext, none, p, mass = extract_then_grade(grader, questions, golds, responses, batch, extract_batch)
    p_first, _ = grader.grade(questions, golds, [first_sentence(r) for r in responses], batch)
    p_full, _ = grader.grade(questions, golds, responses, batch)  # version 2's metric: a diagnostic only
    return pd.DataFrame({
        "p_correct": p, "correct": (p >= 0.5).astype(float), "verdict_mass": mass, "extracted": ext,
        "ext_none": none.astype(float),
        "p_first": p_first, "correct_first": (p_first >= 0.5).astype(float),
        "p_full": p_full, "correct_full": (p_full >= 0.5).astype(float),
        "ext_em": [0.0 if n else exact_match(e, g) for e, g, n in zip(ext, golds, none)],
        "ext_f1": [0.0 if n else f1_score(e, g) for e, g, n in zip(ext, golds, none)],
    })


def questions_of(out: Path, split: str) -> pd.DataFrame:
    return pd.read_parquet(out / "pools" / f"{split}.parquet", columns=["id", "question", "answer"]).set_index("id")


# ---------------------------------------------------------------------------------------------


def known_slice(path: Path, start: int, count: int) -> pd.DataFrame:
    """Rows [start, start + count) of experiment 1's dev cache (HotpotQA dev file order), with question and gold."""
    known = pd.read_parquet(path, columns=["id", "question", "answer", "large_pred", "large_em", "large_f1"])
    return known.iloc[start: start + count].reset_index(drop=True)


def passage_dumps(out: Path, split: str, k: int = 5) -> pd.DataFrame:
    from stair.pool_answer import top_passages

    pools = pd.read_parquet(out / "pools" / f"{split}.parquet").merge(
        pd.read_parquet(out / "scores" / f"{split}.parquet", columns=["id", "nanojev"]), on="id")
    return pd.DataFrame({"id": pools["id"], "question": pools["question"], "answer": pools["answer"],
                         "response": [" ".join(top_passages(r, r["nanojev"], k)) for _, r in pools.iterrows()]})


def check_slice(grader: Grader, known: pd.DataFrame, dumps: pd.DataFrame, batch: int) -> tuple[dict, pd.DataFrame]:
    ek, nk, pk, mk = extract_then_grade(grader, known["question"], known["answer"], known["large_pred"], batch)
    ed, nd, pd_, md = extract_then_grade(grader, dumps["question"], dumps["answer"], dumps["response"], batch)
    # self-check on the graded short answers ("none" is not graded, so it is left out)
    wk = grader.written_verdicts(known["question"][~nk], known["answer"][~nk], [e for e, n in zip(ek, nk) if not n])
    wd = grader.written_verdicts(dumps["question"][~nd], dumps["answer"][~nd], [e for e, n in zip(ed, nd) if not n])
    written = np.concatenate([wk, wd])
    scored = np.concatenate([pk[~nk], pd_[~nd]]) >= 0.5
    parsed = ~np.isnan(written)
    em1, f10 = known["large_em"].to_numpy() == 1, known["large_f1"].to_numpy() == 0
    graded_mass = np.concatenate([mk[~nk], md[~nd]])
    stats = {
        "known_n": int(len(known)), "known_em1_n": int(em1.sum()), "known_f10_n": int(f10.sum()),
        "extract_faithful": float(np.mean([exact_match(e, r) for e, r in zip(ek, known["large_pred"])])),
        "known_extracted_none": float(nk.mean()),
        "agree_em1_correct": float((pk[em1] >= 0.5).mean()), "agree_f10_incorrect": float((pk[f10] < 0.5).mean()),
        "known_graded_correct": float((pk >= 0.5).mean()), "known_em": float(em1.mean()),
        "dump_n": int(len(dumps)), "dump_extracted_none": float(nd.mean()),
        "dump_graded_correct": float((pd_ >= 0.5).mean()),
        "written_parsed": float(parsed.mean()) if len(parsed) else 0.0,
        "written_agreement": float((scored[parsed] == (written[parsed] == 1.0)).mean()) if parsed.any() else 0.0,
        "verdict_mass_mean": float(graded_mass.mean()) if len(graded_mass) else 0.0,
    }
    dis = known.assign(extracted=ek, p_correct=pk)
    dis = dis[(em1 & (pk < 0.5)) | (f10 & (pk >= 0.5))]
    return stats, dis[["id", "question", "answer", "large_pred", "extracted", "large_em", "large_f1", "p_correct"]]


def cmd_validate(args) -> None:
    out = args.out
    grader = Grader(args.grader, args.device, args.revision)
    revision = provenance.hub_revision(args.grader, args.revision)
    provenance.record(out / "rescore", "rescore-validate", grader=args.grader, revision=revision, **provenance.environment())
    start = time.perf_counter()
    gate, gate_dis = check_slice(grader, known_slice(args.known, args.known_start, args.known_count),
                                 passage_dumps(out, args.dump_split), args.batch)
    report = {"grader": args.grader, "revision": revision,
              "gate": {"known": f"{args.known} rows {args.known_start}-{args.known_start + args.known_count - 1}",
                       "dumps": f"{args.dump_split} pools", **gate},
              "thresholds": {"min_agree_em1": MIN_AGREE_EM1, "min_agree_f10": MIN_AGREE_F10,
                             "max_dump_correct": MAX_DUMP_CORRECT, "min_gen_agreement": MIN_GEN_AGREEMENT,
                             "min_verdict_mass": MIN_VERDICT_MASS, "min_extract_faithful": MIN_EXTRACT_FAITHFUL},
              "method": "v3: extract, then grade the extracted answer"}
    if args.diagnostic_start is not None:  # the slice the prompt was diagnosed on: reported, never gating
        diag, _ = check_slice(grader, known_slice(args.known, args.diagnostic_start, args.known_count),
                              passage_dumps(out, "test"), args.batch)
        report["diagnostic_not_gating"] = {"known": f"rows {args.diagnostic_start}-{args.diagnostic_start + args.known_count - 1}",
                                           "dumps": "test pools", **diag}
    report["seconds"] = time.perf_counter() - start
    report["passed"] = bool(gate["agree_em1_correct"] >= MIN_AGREE_EM1 and gate["agree_f10_incorrect"] >= MIN_AGREE_F10
                            and gate["dump_graded_correct"] <= MAX_DUMP_CORRECT
                            and gate["written_agreement"] >= MIN_GEN_AGREEMENT
                            and gate["verdict_mass_mean"] >= MIN_VERDICT_MASS
                            and gate["extract_faithful"] >= MIN_EXTRACT_FAITHFUL)
    gate_dis.head(40).to_csv(out / "rescore" / "validation-disagreements.csv", index=False)
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
                 f"lenient {merged['lenient'].mean():.3f} | graded (extracted) {merged['correct'].mean():.3f} "
                 f"first sentence {merged['correct_first'].mean():.3f} whole (v2) {merged['correct_full'].mean():.3f} | "
                 f"extracted EM {merged['ext_em'].mean():.3f} F1 {merged['ext_f1'].mean():.3f} none {merged['ext_none'].mean():.3f}")


def cmd_summary(args) -> None:
    rows = []
    for split_dir in sorted((args.out / "rescore").iterdir()):
        if not split_dir.is_dir():
            continue
        for f in sorted(split_dir.glob("*.parquet")):
            d = pd.read_parquet(f)
            units = pd.read_parquet(args.out / "answers" / split_dir.name / f.name, columns=["units"])["units"].mean()
            get = lambda c: d[c].mean() if c in d else float("nan")
            rows.append({"split": split_dir.name, "cell": f.stem, "n": len(d), "lenient": get("lenient"),
                         "graded_correct": get("correct"), "graded_first_sentence": get("correct_first"),
                         "graded_whole_v2": get("correct_full"), "ext_em": get("ext_em"), "ext_f1": get("ext_f1"),
                         "ext_none": get("ext_none"), "verdict_mass": get("verdict_mass"), "units": units})
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
                        help="validate: experiment 1's dev cache with short answers and official EM/F1 (dev file order)")
    parser.add_argument("--known-start", type=int, default=1000, help="validate: first row of the fresh gate slice")
    parser.add_argument("--known-count", type=int, default=500)
    parser.add_argument("--dump-split", default="calib", help="validate: pools for the pasted-passage control")
    parser.add_argument("--diagnostic-start", type=int, default=500,
                        help="validate: also report (not gate on) this slice, the one the prompt was diagnosed on")
    parser.add_argument("--skip-validation-check", action="store_true", help="run: do not require a passed validation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    (args.out / "rescore").mkdir(parents=True, exist_ok=True)
    {"validate": cmd_validate, "run": cmd_run, "summary": cmd_summary}[args.step](args)


if __name__ == "__main__":
    main()
