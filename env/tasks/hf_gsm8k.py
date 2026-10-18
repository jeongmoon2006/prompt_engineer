"""Second domain from Hugging Face: GSM8K with a calculator tool.

Loads openai/gsm8k (main) with `datasets`, caches under env/tasks/hf/, fixes split and seed,
and writes env/tasks/gsm8k_v1.jsonl (<=300 train from the HF train split, <=100 held-out from the
HF test split). Ground truth = the number after '####'. Verification stays programmatic.

Run:  python -m env.tasks.hf_gsm8k
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from shared.schemas import Task

TASK_DESCRIPTION = (
    "Solve the grade-school math word problem. You may use the calculator tool for arithmetic, at "
    "most {n_max} times. Answer with one JSON object with the single key \"answer\" holding the final number."
)
CALCULATOR = {"name": "calculator", "input_schema": {
    "type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"], "additionalProperties": False}}


def parse_answer(answer_field: str) -> float:
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_field)
    if not m:
        raise ValueError(f"no #### answer in {answer_field[:80]!r}")
    return float(m.group(1).replace(",", ""))


def to_record(ex: dict[str, Any], idx: int, split: str, n_max: int) -> dict[str, Any]:
    task = Task(task_id=f"gsm8k-{idx:04d}", description=TASK_DESCRIPTION.format(n_max=n_max), context={
        "domain": "gsm8k", "input_text": ex["question"].strip(), "input_label": "Problem",
        "tool": CALCULATOR, "n_max_tool_calls": n_max, "output_fields": ["answer"],
    })
    return {"task": task.model_dump(), "ground_truth": {"answer": parse_answer(ex["answer"])}, "split": split}


def build(seed: int, n_train: int, n_heldout: int, n_max: int, cache_dir: Path, dataset: str, config: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset, config, cache_dir=str(cache_dir))
    rng = random.Random(seed)
    tr_idx = list(range(len(ds["train"])))
    te_idx = list(range(len(ds["test"])))
    rng.shuffle(tr_idx)
    rng.shuffle(te_idx)
    recs = [to_record(ds["train"][i], k, "train", n_max) for k, i in enumerate(tr_idx[:n_train])]
    recs += [to_record(ds["test"][i], n_train + k, "heldout", n_max) for k, i in enumerate(te_idx[:n_heldout])]
    return recs


def main() -> None:
    import yaml

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    hf, dom = cfg["hf"], cfg["domains"]["gsm8k"]
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=root / "tasks" / dom["file"])
    a = p.parse_args()
    cache = root.parent / hf["cache_dir"] if not Path(hf["cache_dir"]).is_absolute() else Path(hf["cache_dir"])
    recs = build(hf["seed"], hf["n_train"], hf["n_heldout"], dom["n_max_tool_calls"], cache, hf["dataset"], hf["config"])
    with a.out.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(recs)} gsm8k tasks -> {a.out}")


if __name__ == "__main__":
    main()
