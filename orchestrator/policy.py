"""Meta-policy pi_phi: small open LLM + LoRA that writes the k slot prompts.

- Base model frozen; only LoRA params train (kept in fp32 as master weights).
- pi_ref is the same model with the adapter disabled (`with model.disable_adapter():`).
- Mixed precision: fp16 autocast on CUDA (Turing has no bf16). Base weights stay fp32 by default
  (0.5B x 4 B = 2 GB fits in 6 GB) which avoids Qwen2.5's known fp16 overflow; set
  base_dtype: float16 to trade stability for memory.
- Output format: one <slot>...</slot> block per slot. Malformed outputs are rejected by `parse_bundle`.
"""

from __future__ import annotations

import os
import re
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from shared.schemas import PromptBundle

META_PROMPT_TEMPLATE = """You write the prompts for an automated agent. The agent is a frozen language model. Its input is {task_description}. It must output one JSON object with the fields {fields}. It can call a tool at most {n_max} times; every tool call and every token costs budget. The agent is scored on: exact correctness of the fields, output format (full credit only for a bare JSON object with no code fences or prose), using as few tool calls as possible, and total tokens used.

The agent has {k} prompt slots:
{slot_descriptions}

Here are the current prompts:

{baseline_block}

Write an improved set of prompts for all {k} slots. Keep each slot concise. Output exactly one block per slot using the same tags, in the same order, and nothing else."""

SLOT_DESCRIPTIONS = {
    "system": "system: the agent's system prompt",
    "planner": "planner: instruction placed in the user turn before the notebook entry",
    "tool_policy": "tool_policy: the description of the tool shown to the agent",
}


def format_bundle_block(slots: dict[str, str], order: list[str]) -> str:
    return "\n".join(f"<{name}>\n{slots[name].strip()}\n</{name}>" for name in order)


@dataclass
class RejectInfo:
    reason: str


def parse_bundle(text: str, slot_names: list[str], max_chars: int) -> PromptBundle | RejectInfo:
    """Parse `<slot>...</slot>` blocks. Any malformation -> RejectInfo (reward -1 downstream)."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return RejectInfo("non_utf8")
    if not text.strip():
        return RejectInfo("empty")
    slots: dict[str, str] = {}
    for name in slot_names:
        found = re.findall(rf"<{name}>(.*?)</{name}>", text, flags=re.DOTALL)
        if len(found) != 1:
            return RejectInfo(f"tag_count_{name}={len(found)}")
        body = found[0].strip()
        if not body:
            return RejectInfo(f"empty_slot_{name}")
        if len(body) > max_chars:
            return RejectInfo(f"too_long_{name}={len(body)}")
        if re.search(r"</?\w+>", body):  # nested / stray tags inside a slot
            return RejectInfo(f"stray_tag_in_{name}")
        slots[name] = body
    leftover = re.sub(r"<(\w+)>.*?</\1>", "", text, flags=re.DOTALL).strip()
    if len(leftover) > 40:  # tolerate a few stray chars, reject chatter around the blocks
        return RejectInfo("extra_text")
    return PromptBundle(bundle_id=f"grpo-{uuid.uuid4().hex[:12]}", slots=slots)


def select_device() -> torch.device:
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetaPolicy:
    def __init__(self, cfg: dict[str, Any], baseline: dict[str, str], task_info: dict[str, Any] | None = None,
                 device: torch.device | None = None) -> None:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.slot_names: list[str] = list(cfg["slots"])
        self.baseline = baseline
        self.device = device or select_device()
        self.use_amp = self.device.type == "cuda"
        base_dtype = {"float32": torch.float32, "float16": torch.float16}[cfg.get("base_dtype", "float32")]
        if self.device.type == "cpu":
            base_dtype = torch.float32

        name = cfg["base_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        load_kw: dict[str, Any] = {"torch_dtype": base_dtype}
        if cfg.get("load_in_4bit", False):  # opt-in QLoRA (bitsandbytes), needed for 1.5B on 6 GB
            from transformers import BitsAndBytesConfig
            load_kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                                                bnb_4bit_quant_type="nf4")
            load_kw["device_map"] = {"": 0}
        base = AutoModelForCausalLM.from_pretrained(name, **load_kw)
        if not cfg.get("load_in_4bit", False):
            base.to(self.device)
        base.config.use_cache = True
        if cfg.get("gradient_checkpointing", True):
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.enable_input_require_grads()
        lcfg = cfg["lora"]
        self.model = get_peft_model(base, LoraConfig(
            r=int(lcfg["r"]), lora_alpha=int(lcfg["alpha"]), lora_dropout=float(lcfg["dropout"]),
            target_modules=list(lcfg["target_modules"]), bias="none", task_type="CAUSAL_LM",
        ))
        for p in self.model.parameters():  # LoRA master weights in fp32
            if p.requires_grad:
                p.data = p.data.float()
        self.model.train()

        info = task_info or {}
        self.meta_prompt = META_PROMPT_TEMPLATE.format(
            task_description=info.get("description", "a short task description"),
            fields=", ".join(info.get("fields", ["answer"])),
            n_max=info.get("n_max", 4), k=len(self.slot_names),
            slot_descriptions="\n".join(f"- {SLOT_DESCRIPTIONS.get(s, s)}" for s in self.slot_names),
            baseline_block=format_bundle_block(baseline, self.slot_names),
        )
        chat = [{"role": "user", "content": self.meta_prompt}]
        self.prompt_text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        self.prompt_ids = self.tokenizer(self.prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

    # ------------------------------------------------------------------ helpers
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.float16) if self.use_amp else nullcontext()

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample(self, n: int, temperature: float | None = None) -> list[dict[str, Any]]:
        """Sample n completions from pi_phi. Returns [{text, completion_ids, bundle|None, reject_reason|None}]."""
        temperature = float(temperature if temperature is not None else self.cfg.get("temperature", 1.0))
        self.model.eval()
        if self.cfg.get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_disable()  # incompatible with use_cache during generation
        self.model.config.use_cache = True
        inp = self.prompt_ids.unsqueeze(0).to(self.device)
        attn = torch.ones_like(inp)
        with self.autocast():
            out = self.model.generate(
                input_ids=inp, attention_mask=attn, do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
                max_new_tokens=int(self.cfg.get("max_new_tokens", 512)), num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
            )
        comps = out[:, inp.shape[1]:].cpu()
        results = []
        for row in comps:
            # cut at the first EOS (kept, so the policy learns to stop) or strip trailing padding
            ids = _strip_padding(row, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id)
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            parsed = parse_bundle(text, self.slot_names, int(self.cfg.get("max_slot_chars", 1500)))
            results.append({
                "text": text, "completion_ids": ids,
                "bundle": parsed if isinstance(parsed, PromptBundle) else None,
                "reject_reason": parsed.reason if isinstance(parsed, RejectInfo) else None,
            })
        self.model.train()
        if self.cfg.get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if self.device.type == "cuda":
            torch.cuda.empty_cache()  # free the generation KV cache before training passes
        return results

    # ------------------------------------------------------------------ log-probs
    def completion_logprobs(self, completion_ids: list[torch.Tensor], ref: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token log-probs of each completion given the fixed meta-prompt.

        Returns (logp [B, T], mask [B, T]) right-padded over completions. ref=True disables the
        adapter and runs under no_grad (pi_ref).
        """
        B = len(completion_ids)
        T = max(int(c.numel()) for c in completion_ids)
        P = int(self.prompt_ids.numel())
        pad = self.tokenizer.pad_token_id
        seq = torch.full((B, P + T), pad, dtype=torch.long)
        mask = torch.zeros((B, T), dtype=torch.float32)
        for i, c in enumerate(completion_ids):
            seq[i, :P] = self.prompt_ids
            seq[i, P:P + c.numel()] = c
            mask[i, :c.numel()] = 1.0
        attn = torch.ones_like(seq)
        attn[:, P:] = mask.long()
        seq, attn, mask = seq.to(self.device), attn.to(self.device), mask.to(self.device)
        ctx = self.model.disable_adapter() if ref else nullcontext()
        grad_ctx = torch.no_grad() if ref else nullcontext()
        with ctx, grad_ctx, self.autocast():
            out = self.model(input_ids=seq, attention_mask=attn, use_cache=False, logits_to_keep=T + 1)
        logits = out.logits[:, :-1, :]                # predicts positions P .. P+T-1 (fp16 under autocast)
        targets = seq[:, P:]
        # Selective log-softmax, one sequence at a time: avoids materialising a [B, T, V] fp32 log_softmax
        # (V = 152k makes that ~0.5 GB per sequence) which matters on a 6 GB card.
        rows = []
        for b in range(B):
            lb = logits[b].float()
            rows.append(lb.gather(-1, targets[b].unsqueeze(-1)).squeeze(-1) - torch.logsumexp(lb, dim=-1))
            del lb
        logp = torch.stack(rows, dim=0)
        return logp * mask, mask

    # ------------------------------------------------------------------ persistence
    def save_adapter(self, path: str) -> None:
        self.model.save_pretrained(path)

    def load_adapter(self, path: str) -> None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        state = load_file(os.path.join(path, "adapter_model.safetensors"))
        set_peft_model_state_dict(self.model, state)


def _strip_padding(row: torch.Tensor, pad_id: int, eos_id: int) -> torch.Tensor:
    ids = row.tolist()
    # cut after the first EOS (inclusive); otherwise strip trailing pads
    if eos_id in ids:
        ids = ids[: ids.index(eos_id) + 1]
    else:
        while ids and ids[-1] == pad_id:
            ids.pop()
    return torch.tensor(ids, dtype=torch.long)
