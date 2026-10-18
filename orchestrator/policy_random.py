"""Phase 1 placeholder policy: G random perturbations of the baseline prompts.

Exists only to exercise the loop (and later as baseline B1). Replaced by the GRPO
meta-policy in Phase 2; kept because Phase 3 compares against it.
"""

from __future__ import annotations

import random
import re
import uuid

from shared.schemas import PromptBundle

SYNONYMS: dict[str, list[str]] = {
    "careful": ["meticulous", "precise", "diligent"],
    "read": ["examine", "study", "inspect"],
    "identify": ["determine", "find", "extract"],
    "respond": ["answer", "reply"],
    "single": ["one", "exactly one"],
    "retrieve": ["fetch", "obtain", "get"],
    "stated": ["given", "present", "written"],
    "entry": ["record", "note"],
    "assistant": ["helper", "agent"],
    "only": ["solely", "exclusively"],
    "produce": ["generate", "output", "emit"],
}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def perturb_text(text: str, rng: random.Random, p_swap: float = 0.3, shuffle: bool = True) -> str:
    sentences = [s for s in _SENT_SPLIT.split(" ".join(text.split())) if s]
    if shuffle and len(sentences) > 1 and rng.random() < 0.5:
        rng.shuffle(sentences)
    out = []
    for s in sentences:
        words = s.split(" ")
        for i, w in enumerate(words):
            key = re.sub(r"[^a-z]", "", w.lower())
            if key in SYNONYMS and rng.random() < p_swap:
                rep = rng.choice(SYNONYMS[key])
                if w[:1].isupper():
                    rep = rep.capitalize()
                words[i] = re.sub(re.escape(key), rep, w, flags=re.IGNORECASE)
        out.append(" ".join(words))
    return " ".join(out)


class RandomPerturbationPolicy:
    def __init__(self, baseline: dict[str, str], seed: int = 0, p_swap: float = 0.3) -> None:
        self.baseline = baseline
        self.rng = random.Random(seed)
        self.p_swap = p_swap

    def sample(self, group_size: int) -> list[PromptBundle]:
        bundles = []
        for _ in range(group_size):
            slots = {k: perturb_text(v, self.rng, self.p_swap) for k, v in self.baseline.items()}
            bundles.append(PromptBundle(bundle_id=f"rnd-{uuid.UUID(int=self.rng.getrandbits(128)).hex[:12]}", slots=slots))
        return bundles
