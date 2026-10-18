"""GRPO building blocks (pure torch; no model code here).

  A_i = (R_i - mean(R)) / (std(R) + eps)                      within each group
  L   = -mean_tokens[ min(r*A, clip(r, 1-eps, 1+eps)*A) - beta * KL ]
  r   = exp(logp_new - logp_old)                               per token
  KL  = exp(logp_ref - logp_new) - (logp_ref - logp_new) - 1   per-token k3 estimator (>= 0, unbiased)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """rewards: [G] -> standardized advantages [G]. A single-element or constant group gives zeros."""
    if rewards.numel() < 2:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps)


@dataclass
class GRPOStats:
    loss: float
    policy_loss: float
    kl: float
    ratio_mean: float
    ratio_min: float
    ratio_max: float
    clip_frac: float
    n_tokens: int


def grpo_loss(
    logp_new: torch.Tensor,   # [B, T] log-probs of completion tokens under current pi_phi (with grad)
    logp_old: torch.Tensor,   # [B, T] under pi_phi_old (no grad); equals logp_new.detach() when mu = 1
    logp_ref: torch.Tensor,   # [B, T] under pi_ref (adapter disabled, no grad)
    advantages: torch.Tensor, # [B]
    mask: torch.Tensor,       # [B, T] 1 for real completion tokens, 0 for padding
    clip_eps: float = 0.2,
    beta: float = 0.04,
) -> tuple[torch.Tensor, GRPOStats]:
    mask = mask.to(logp_new.dtype)
    adv = advantages.to(logp_new.dtype).unsqueeze(1)
    log_ratio = logp_new - logp_old
    ratio = torch.exp(log_ratio)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    surrogate = torch.minimum(unclipped, clipped)
    d = logp_ref - logp_new
    kl = torch.exp(d) - d - 1.0
    per_token = -(surrogate - beta * kl)
    tok_per_seq = mask.sum(dim=1).clamp(min=1.0)
    per_seq = (per_token * mask).sum(dim=1) / tok_per_seq
    loss = per_seq.mean()

    with torch.no_grad():
        n_tok = mask.sum()
        m = mask.bool()
        is_clipped = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)) & m
        stats = GRPOStats(
            loss=float(loss),
            policy_loss=float((-(surrogate * mask).sum(dim=1) / tok_per_seq).mean()),
            kl=float((kl * mask).sum() / n_tok.clamp(min=1)),
            ratio_mean=float(ratio[m].mean()) if n_tok > 0 else 1.0,
            ratio_min=float(ratio[m].min()) if n_tok > 0 else 1.0,
            ratio_max=float(ratio[m].max()) if n_tok > 0 else 1.0,
            clip_frac=float(is_clipped.sum() / n_tok.clamp(min=1)),
            n_tokens=int(n_tok),
        )
    return loss, stats


class GRPOUpdater:
    """One optimizer step per batch of groups: micro-batched forward/backward with fp16 autocast +
    GradScaler on CUDA, gradient accumulation, grad-norm clipping, VRAM guard (auto-halves the
    micro-batch above `vram_limit_gb` or on OOM)."""

    def __init__(self, policy, cfg: dict) -> None:
        self.policy = policy
        self.cfg = cfg
        self.clip_eps = float(cfg["clip_eps"])
        self.beta = float(cfg["kl_beta"])
        self.adv_eps = float(cfg["adv_eps"])
        self.micro_batch = int(cfg["micro_batch"])
        self.vram_limit_gb = float(cfg.get("vram_limit_gb", 5.5))
        self.grad_clip = float(cfg["grad_clip_norm"])
        self.opt = torch.optim.AdamW(policy.trainable_parameters(), lr=float(cfg["lr"]), weight_decay=0.0)
        self.cuda = policy.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.cuda)

    def update(self, completions: list[torch.Tensor], advantages: list[float]) -> dict:
        """completions: completion token ids per trajectory; advantages: standardized, same order."""
        n = len(completions)
        if n == 0:
            return {"skipped": True}
        for attempt in range(4):
            try:
                return self._update_once(completions, advantages, n)
            except torch.cuda.OutOfMemoryError:
                self.opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if self.micro_batch == 1:
                    raise
                self.micro_batch = max(1, self.micro_batch // 2)
        raise RuntimeError("GRPO update failed after repeated OOM")

    def _update_once(self, completions, advantages, n) -> dict:
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()
        self.opt.zero_grad(set_to_none=True)
        agg = {"loss": 0.0, "policy_loss": 0.0, "kl": 0.0, "clip_frac": 0.0, "ratio_mean": 0.0, "n_tokens": 0}
        ratio_min, ratio_max = float("inf"), float("-inf")
        order = sorted(range(n), key=lambda i: int(completions[i].numel()))  # length-bucket to cut padding
        mb = self.micro_batch
        for s in range(0, n, mb):
            idx = order[s:s + mb]
            ids = [completions[i] for i in idx]
            adv = torch.tensor([advantages[i] for i in idx], dtype=torch.float32, device=self.policy.device)
            logp_ref, _ = self.policy.completion_logprobs(ids, ref=True)
            logp_new, mask = self.policy.completion_logprobs(ids, ref=False)
            loss, st = grpo_loss(logp_new, logp_new.detach(), logp_ref, adv, mask, self.clip_eps, self.beta)
            self.scaler.scale(loss * (len(idx) / n)).backward()
            w = len(idx) / n
            for k in ("loss", "policy_loss", "kl", "clip_frac", "ratio_mean"):
                agg[k] += getattr(st, k) * w
            agg["n_tokens"] += st.n_tokens
            ratio_min, ratio_max = min(ratio_min, st.ratio_min), max(ratio_max, st.ratio_max)
            del logp_ref, logp_new, mask, loss
        self.scaler.unscale_(self.opt)
        gn = torch.nn.utils.clip_grad_norm_(self.policy.trainable_parameters(), self.grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / 1e9 if self.cuda else 0.0
        mb_before = self.micro_batch
        if self.cuda and peak > self.vram_limit_gb and self.micro_batch > 1:
            self.micro_batch = max(1, self.micro_batch // 2)
        if self.cuda:
            torch.cuda.empty_cache()
        return {**agg, "ratio_min": ratio_min, "ratio_max": ratio_max, "grad_norm": float(gn),
                "peak_vram_gb": round(peak, 3), "micro_batch": mb_before, "micro_batch_next": self.micro_batch,
                "n_sequences": n, "lr": self.opt.param_groups[0]["lr"], "scale": float(self.scaler.get_scale()) if self.cuda else 1.0}
