"""GRPO math on hand-built tensors (CPU)."""

import math

import pytest

torch = pytest.importorskip("torch")

from orchestrator.grpo import group_advantages, grpo_loss


def test_group_advantages_standardized() -> None:
    r = torch.tensor([1.0, 2.0, 3.0, 4.0])
    a = group_advantages(r)
    assert math.isclose(float(a.mean()), 0.0, abs_tol=1e-6)
    assert math.isclose(float(a.std(unbiased=False)), 1.0, rel_tol=1e-3)
    assert torch.all(group_advantages(torch.tensor([0.5, 0.5, 0.5])) == 0)
    assert torch.all(group_advantages(torch.tensor([0.7])) == 0)


def test_loss_zero_grad_direction_and_kl() -> None:
    torch.manual_seed(0)
    B, T = 4, 6
    logp_new = (-torch.rand(B, T) - 0.1).requires_grad_(True)
    logp_old = logp_new.detach().clone()
    logp_ref = logp_new.detach().clone()
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    mask = torch.ones(B, T)
    mask[3, 4:] = 0
    loss, st = grpo_loss(logp_new, logp_old, logp_ref, adv, mask, clip_eps=0.2, beta=0.04)
    # mu = 1: ratio is exactly 1, nothing clipped, KL to an identical ref is 0.
    assert math.isclose(st.ratio_mean, 1.0, abs_tol=1e-6) and st.clip_frac == 0.0
    assert math.isclose(st.kl, 0.0, abs_tol=1e-6)
    assert math.isclose(float(loss), -float(adv.mean()), abs_tol=1e-5)  # -mean(ratio*A) with ratio=1
    loss.backward()
    g = logp_new.grad
    # gradient pushes up log-probs of positive-advantage sequences and down for negative ones
    assert torch.all(g[0] < 0) and torch.all(g[1] > 0)
    assert torch.all(g[3, 4:] == 0)  # masked tokens get no gradient
    assert st.n_tokens == B * T - 2


def test_clipping_and_kl_positive() -> None:
    logp_old = torch.full((2, 3), -1.0)
    logp_new = torch.tensor([[-0.5, -0.5, -0.5], [-1.6, -1.6, -1.6]])  # ratios e^0.5 and e^-0.6
    logp_ref = torch.full((2, 3), -1.2)
    adv = torch.tensor([1.0, 1.0])
    loss, st = grpo_loss(logp_new, logp_old, logp_ref, adv, torch.ones(2, 3), clip_eps=0.2, beta=0.0)
    assert st.clip_frac == 1.0
    # clipped surrogate: min(r*A, clip(r)*A) -> row0: 1.2, row1: e^-0.6
    assert math.isclose(float(loss), -(1.2 + math.exp(-0.6)) / 2, rel_tol=1e-5)
    _, st2 = grpo_loss(logp_new, logp_old, logp_ref, adv, torch.ones(2, 3), beta=0.04)
    assert st2.kl > 0
