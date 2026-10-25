"""MetaPolicy on a tiny random Qwen2 model (CPU): parsing guards, sampling shapes, log-probs, ref = adapter off."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from orchestrator.policy import MetaPolicy, format_bundle_block, parse_bundle
from shared.schemas import PromptBundle

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
    long = format_bundle_block({k: "y" * 1000 for k in SLOTS}, SLOTS)   # each slot under 1500, total 3000
    assert isinstance(parse_bundle(long, SLOTS, 1500), PromptBundle)
    assert isinstance(parse_bundle(long, SLOTS, 1500, max_total_chars=4000), PromptBundle)
    assert parse_bundle(long, SLOTS, 1500, max_total_chars=2500).reason == "too_long_bundle=3000"


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A tiny random Qwen2 (built locally from a config) + the real Qwen2.5 tokenizer/chat template."""
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    d = tmp_path_factory.mktemp("tiny-qwen2")
    try:  # needs the Qwen2.5 tokenizer files (HF hub cache); skip rather than fail when offline
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Qwen2.5 tokenizer unavailable offline: {type(e).__name__}")
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, tie_word_embeddings=True)
    torch.manual_seed(0)
    Qwen2ForCausalLM(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return str(d)


@pytest.fixture(scope="module")
def tiny_policy(tiny_model_dir: str) -> MetaPolicy:
    cfg = {"base_model": tiny_model_dir, "base_dtype": "float32",
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
