"""Paragraph scoring and the evidence features the router sees.

`LexicalScorer` needs no model and is for smoke tests. `CrossEncoderScorer` loads a nano-jev
checkpoint and uses its `relevance` and `sufficient` decisions.
"""

import json
import math
import re
from pathlib import Path

import numpy as np

STOPWORDS = frozenset(
    "a an the of in on at to for from by with and or but is are was were be been being do does did "
    "what which who whom whose when where why how that this these those it its as than both same "
    "has have had not no yes".split()
)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def content_words(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in STOPWORDS}


def paragraph_text(title: str, sentences: list[str]) -> str:
    return f"{title}: " + " ".join(s.strip() for s in sentences)


class LexicalScorer:
    """Fraction of the question's content words that appear in each paragraph."""

    name = "lexical"

    def score(self, question: str, paragraphs: list[str]) -> np.ndarray:
        q = content_words(question)
        if not q:
            return np.zeros(len(paragraphs))
        return np.array([len(q & content_words(p)) / len(q) for p in paragraphs])

    def sufficiency(self, question: str, paragraphs: list[str]) -> float:
        return float("nan")


class CrossEncoderScorer:
    """nano-jev cross-encoder: one logit per (question + option, context) pair."""

    name = "cross-encoder"
    RELEVANCE = ("How relevant is this passage to the query: {q}", ["irrelevant", "partially relevant", "directly answers"])
    SUFFICIENT = ("Does the context contain enough information to answer: {q}", ["yes", "no"])

    def __init__(self, path: str, device: str | None = None, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
        self.max_length = max_length
        self.temperatures = {"relevance": 1.0, "sufficient": 1.0}
        calib = Path(path) / "calibration.json"
        if calib.exists():
            self.temperatures.update(json.loads(calib.read_text()))
        self.truncated = 0
        self.pairs_seen = 0

    def _decide(self, decision: str, question: str, states: list[str]) -> np.ndarray:
        """Return [len(states), num_options] temperature-scaled probabilities."""
        template, options = self.RELEVANCE if decision == "relevance" else self.SUFFICIENT
        text = template.format(q=question)
        firsts = [f"question: {text} option: {opt}" for _ in states for opt in options]
        seconds = [s for s in states for _ in options]
        enc = self.tokenizer(firsts, seconds, truncation="only_second", max_length=self.max_length, padding=True, return_tensors="pt")
        full_lengths = [len(ids) for ids in self.tokenizer(firsts, seconds)["input_ids"]]
        self.truncated += sum(n > self.max_length for n in full_lengths)
        self.pairs_seen += len(full_lengths)
        with self.torch.no_grad():
            logits = self.model(**enc.to(self.device)).logits[:, 0].float().cpu().numpy()
        logits = logits.reshape(len(states), len(options)) / self.temperatures[decision]
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        return probs / probs.sum(axis=1, keepdims=True)

    def score(self, question: str, paragraphs: list[str]) -> np.ndarray:
        """P(not irrelevant) per paragraph."""
        return 1.0 - self._decide("relevance", question, paragraphs)[:, 0]

    def sufficiency(self, question: str, paragraphs: list[str]) -> float:
        """P(yes) that the given paragraphs together are enough to answer."""
        return float(self._decide("sufficient", question, ["\n".join(paragraphs)])[0, 0])


def make_scorer(cfg: dict):
    kind = cfg.get("kind", "lexical")
    if kind == "lexical":
        return LexicalScorer()
    if kind == "cross-encoder":
        return CrossEncoderScorer(cfg["path"], device=cfg.get("device"))
    raise ValueError(f"unknown scorer kind: {kind}")


def evidence_features(scores: np.ndarray, sufficiency: float) -> dict[str, float]:
    """Features of the paragraph score distribution. Uses no gold information."""
    s = np.sort(scores)[::-1]
    dist = scores / scores.sum() if scores.sum() > 0 else np.full(len(scores), 1 / len(scores))
    entropy = -sum(p * math.log(p) for p in dist if p > 0)
    return {
        "ev_top1": float(s[0]),
        "ev_top2_sum": float(s[:2].sum()),
        "ev_margin_2_3": float(s[1] - s[2]) if len(s) > 2 else 0.0,
        "ev_entropy": float(entropy),
        "ev_sufficiency": sufficiency,
    }
