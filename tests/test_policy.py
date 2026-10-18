"""MetaPolicy on a tiny random Qwen2 model (CPU): parsing guards, sampling shapes, log-probs, ref = adapter off."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from orchestrator.policy import MetaPolicy, RejectInfo, format_bundle_block, parse_bundle  # noqa: E402
from shared.schemas import PromptBundle  # noqa: E402

SLOTS = ["system", "planner", "tool_policy"]
BASE = {"system": "Be careful.", "planner": "Read then answer.", "tool_policy": "Use lookup sparingly."}


def test_parse_bundle_roundtrip_and_guards() -> None:
    text = format_bundle_block(BASE, SLOTS)
    b = parse_bundle(text, SLOTS, 1500)
    assert isinstance(b, PromptBundle) and b.slots == BASE
    assert parse_bundle("", SLOTS, 1500).reason == "empty"
    assert parse_bundle(text.replace("</planner>", ""), SLOTS, 1500).reason.startswith("tag_count_planner")
    assert parse_bundle(text + text, SLOTS, 1500).reason.startswith("tag_count_")
    assert parse_bundle(text.replace("Be careful.", "x" * 2000), SLOTS, 1500).reason.startswith("too_long_system")
    assert parse_bundle(text.replace("Be careful.", "   "), SLOTS, 1500).reason == "empty_slot_system"
    assert parse_bundle(text.replace("Be careful.", "a <b> c"), SLOTS, 1500).reason == "stray_tag_in_system"
    assert parse_bundle("Sure! Here are much better prompts for you, hope they help:\n" + text, SLOTS, 1500).reason == "extra_text"
    assert isinstance(parse_bundle("ok\n" + text, SLOTS, 1500), PromptBundle)  # small chatter tolerated


@pytest.fixture(scope="module")
def tiny_policy() -> MetaPolicy:
    cfg = {"base_model": "hf-internal-testing/tiny-random-Qwen2ForCausalLM", "base_dtype": "float32",
           "gradient_checkpointing": False, "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
           "temperature": 1.0, "max_new_tokens": 12, "max_slot_chars": 1500, "slots": SLOTS}
    return MetaPolicy(cfg, BASE, task_info={"fields": ["a", "b"], "n_max": 2}, device=torch.device("cpu"))


def test_sample_shapes_and_rejection(tiny_policy: MetaPolicy) -> None:
    out = tiny_policy.sample(3)
    assert len(out) == 3
    for s in out:
        assert s["completion_ids"].ndim == 1 and 1 <= s["completion_ids"].numel() <= 12
        assert (s["bundle"] is None) == (s["reject_reason"] is not None)  # a random model never emits valid tags


def test_logprobs_grad_and_reference(tiny_policy: MetaPolicy) -> None:
    out = tiny_policy.sample(2)
    ids = [s["completion_ids"] for s in out]
    logp, mask = tiny_policy.completion_logprobs(ids)
    assert logp.shape == mask.shape == (2, max(c.numel() for c in ids))
    assert torch.all(logp <= 0) and logp.requires_grad
    ref, _ = tiny_policy.completion_logprobs(ids, ref=True)
    assert not ref.requires_grad
    # LoRA B is zero-initialised, so pi_phi == pi_ref before any update
    assert torch.allclose(logp.detach(), ref, atol=1e-5)
    assert tiny_policy.n_trainable() > 0
    # after a tiny optimizer step the two differ
    (logp * mask).sum().backward()
    with torch.no_grad():
        for p in tiny_policy.trainable_parameters():
            if p.grad is not None:
                p -= 1e-1 * p.grad.sign()
    logp2, _ = tiny_policy.completion_logprobs(ids)
    ref2, _ = tiny_policy.completion_logprobs(ids, ref=True)
    assert torch.allclose(ref2, ref, atol=1e-5)          # reference unaffected by LoRA
    assert not torch.allclose(logp2.detach(), ref2, atol=1e-5)
