"""Answering experts. Each reads the question plus its top-k paragraphs.

`HeuristicExpert` needs no model and exists only to exercise the pipeline. `HFExpert` runs a
Hugging Face causal LM with greedy decoding.
"""

import math
import re
from dataclasses import dataclass

from groupproject.evidence import content_words, tokenize

AUX_VERBS = frozenset("is are was were do does did can could has have had will would".split())

# Two worked examples, chosen on calib questions: small-model F1 0.235 -> 0.466 vs asking for
# "yes/no for yes/no questions", which made Qwen2.5-0.5B answer "Yes" to open questions.
PROMPT = (
    "Answer the question using the paragraphs below. Reply with only the answer, a few words, not a sentence.\n\n"
    "Example: Question: Which city is the birthplace of the author of Dracula? Answer: Dublin\n"
    "Example: Question: Are Paris and Rome both capital cities? Answer: yes\n\n"
    "{context}\n\nQuestion: {question}\nAnswer:"
)


@dataclass
class ExpertOutput:
    answer: str
    cost: int  # prompt + generated tokens (whitespace tokens for the heuristic)
    uncertainty: float  # higher = less sure; mean token entropy for LMs


class HeuristicExpert:
    """Picks the sentence with most question overlap and returns a capitalised phrase from it.

    With hops=1 it then follows that phrase to the paragraph of the same title and answers from
    there, which is how a bridge question is resolved when the paragraph is in its context.
    """

    def __init__(self, name: str, top_k: int | None, hops: int = 0):
        self.name, self.top_k, self.hops = name, top_k, hops

    @staticmethod
    def _best_sentence(words: set[str], paragraphs: list[str]) -> tuple[str, float]:
        best, best_overlap = "", -1.0
        for para in paragraphs:
            for sent in re.split(r"(?<=[.!?])\s+", para):
                overlap = len(words & content_words(sent)) / max(len(words), 1)
                if overlap > best_overlap:
                    best, best_overlap = sent, overlap
        return best, best_overlap

    @staticmethod
    def _new_phrase(sentence: str, known: set[str]) -> str | None:
        for phrase in re.findall(r"[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*", sentence.split(":", 1)[-1]):
            if not content_words(phrase) <= known:
                return phrase
        return None

    def answer(self, question: str, paragraphs: list[str]) -> ExpertOutput:
        cost = len(question.split()) + sum(len(p.split()) for p in paragraphs)
        if tokenize(question)[:1] and tokenize(question)[0] in AUX_VERBS:
            return ExpertOutput("yes", cost, 0.5)

        known = content_words(question)
        sentence, overlap = self._best_sentence(known, paragraphs)
        phrase = self._new_phrase(sentence, known)
        for _ in range(self.hops):
            if phrase is None:
                break
            linked = [p for p in paragraphs if p.startswith(f"{phrase}:")]
            if not linked:
                break
            known |= content_words(phrase)
            phrase = self._new_phrase(linked[0], known) or phrase
        if phrase is None:
            return ExpertOutput(sentence.split(":")[0], cost, 1.0)
        return ExpertOutput(phrase, cost, 1.0 - overlap)


class HFExpert:
    def __init__(self, name: str, top_k: int | None, model: str, device: str | None = None,
                 max_new_tokens: int = 16, dtype: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name, self.top_k, self.max_new_tokens = name, top_k, max_new_tokens
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(self.device).eval()

    def _prompt_ids(self, question: str, paragraphs: list[str]):
        text = PROMPT.format(context="\n\n".join(paragraphs), question=question)
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True, return_tensors="pt", return_dict=True
            )["input_ids"]
        return self.tokenizer(text, return_tensors="pt")["input_ids"]

    def answer(self, question: str, paragraphs: list[str]) -> ExpertOutput:
        input_ids = self._prompt_ids(question, paragraphs).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                input_ids,
                attention_mask=self.torch.ones_like(input_ids),
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_ids = out.sequences[0, input_ids.shape[1]:]
        entropies = []
        for step_scores in out.scores:
            logp = self.torch.log_softmax(step_scores[0].float(), dim=-1)
            entropies.append(float(-(logp.exp() * logp).sum()))
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        answer = text.strip().split("\n")[0].strip().rstrip(".")
        uncertainty = sum(entropies) / len(entropies) if entropies else math.inf
        return ExpertOutput(answer, int(input_ids.shape[1] + len(new_ids)), uncertainty)


def make_expert(cfg: dict):
    kind = cfg.get("kind", "heuristic")
    top_k = cfg.get("top_k")
    if kind == "heuristic":
        return HeuristicExpert(cfg["name"], top_k, hops=cfg.get("hops", 0))
    if kind == "hf":
        return HFExpert(cfg["name"], top_k, cfg["model"], device=cfg.get("device"),
                        max_new_tokens=cfg.get("max_new_tokens", 16), dtype=cfg.get("dtype", "auto"))
    raise ValueError(f"unknown expert kind: {kind}")
