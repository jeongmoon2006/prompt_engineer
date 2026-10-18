"""cost: -lambda * total_tokens / token_budget (a small negative number)."""

from __future__ import annotations


def score_cost(total_tokens: int, token_budget: int, lam: float) -> float:
    if token_budget <= 0:
        return 0.0
    return -lam * max(total_tokens, 0) / token_budget
