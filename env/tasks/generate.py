"""Toy domain task generator: synthetic lab-notebook extraction.

Deterministic given the seed. Writes env/tasks/toy_v1.jsonl with one record per
line: {"task": Task, "ground_truth": {...}, "split": "train"|"heldout"}.

The agent sees only `Task` (description + context). Ground truth stays in this
file and is read by the environment, never sent to the harness.

Run:  python -m env.tasks.generate            (from the repo root)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from shared.schemas import Task

FIELDS = ["sample_id", "temperature_K", "concentration_mM", "method"]
METHODS = ["DLS", "NMR", "UV-Vis", "SAXS", "ITC", "cryo-EM", "FTIR", "HPLC"]
BUFFERS = ["Tris", "HEPES", "PBS", "MES", "phosphate"]
OPERATORS = ["J.K.", "M.R.", "A.S.", "L.W.", "T.N."]
INSTRUMENTS = ["Zetasizer Nano", "Bruker 600", "Cary 60", "Xeuss 3.0", "MicroCal PEAQ", "Titan Krios"]
LOOKUP_FIELDS = ["sample_id", "temperature", "concentration", "method", "operator", "date", "instrument", "buffer"]

TASK_DESCRIPTION = (
    "Extract a structured record from the lab notebook entry. The record has four fields: "
    "sample_id (string), temperature_K (measurement temperature in kelvin, number), "
    "concentration_mM (sample concentration in millimolar, number) and method (measurement "
    "technique abbreviation). Some values are not written in the entry and must be retrieved "
    "with the lookup tool, which may be called at most {n_max} times. Answer with one JSON object."
)


def _fmt_num(x: float) -> str:
    return f"{x:g}"


def _render_temperature(rng: random.Random, temp_k: float) -> str:
    if rng.random() < 0.5:
        return f"{_fmt_num(round(temp_k - 273.15, 2))} °C"
    return f"{_fmt_num(temp_k)} K"


def _render_concentration(rng: random.Random, conc_mm: float) -> str:
    r = rng.random()
    if r < 0.25 and conc_mm < 1.0:
        return f"{_fmt_num(round(conc_mm * 1000, 3))} µM"
    if r < 0.4 and conc_mm >= 100:
        return f"{_fmt_num(round(conc_mm / 1000, 4))} M"
    return f"{_fmt_num(conc_mm)} mM"


def make_task(rng: random.Random, idx: int, n_max: int) -> tuple[Task, dict[str, Any]]:
    sample_id = rng.choice(["S-", "LN-", "BX", "P"]) + f"{rng.randint(1, 9999):04d}" + rng.choice(["", "A", "B", "-r2"])
    temp_k = round(rng.choice([277.15, 283.15, 293.15, 295.15, 298.15, 303.15, 310.15, 318.15, 323.15, 333.15])
                   + rng.choice([0.0, 0.0, 0.5, -0.5, 1.0, 2.0]), 2)
    conc_mm = round(rng.choice([0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 150.0, 250.0])
                    * rng.choice([1.0, 1.0, 1.2, 0.8]), 4)
    method = rng.choice(METHODS)
    gt = {"sample_id": sample_id, "temperature_K": temp_k, "concentration_mM": conc_mm, "method": method}

    # Distractors
    buffer_conc = rng.choice([10, 20, 25, 50, 100])
    buffer = rng.choice(BUFFERS)
    storage_k = rng.choice([253.15, 277.15, 193.15])
    operator = rng.choice(OPERATORS)
    instrument = rng.choice(INSTRUMENTS)
    date = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"

    # Which target fields are missing from the text (must be looked up): 1 or 2 of 4.
    n_missing = rng.choice([1, 1, 2])
    missing = set(rng.sample(FIELDS, n_missing))

    sentences: list[str] = []
    if "sample_id" in missing:
        sentences.append("Sample ID recorded in the instrument metadata.")
    else:
        sentences.append(rng.choice([f"Sample {sample_id} was taken from the freezer stock (stored at {_fmt_num(storage_k)} K).",
                                     f"Worked on {sample_id} today; aliquot thawed from {_fmt_num(storage_k)} K storage."]))
    conc_txt = _render_concentration(rng, conc_mm)
    if "concentration_mM" in missing:
        sentences.append(f"Diluted into {buffer_conc} mM {buffer} buffer, pH 7.4; final sample concentration logged in metadata.")
    else:
        sentences.append(rng.choice([f"Diluted to {conc_txt} in {buffer_conc} mM {buffer} buffer, pH 7.4.",
                                     f"Final concentration {conc_txt} ({buffer_conc} mM {buffer}, pH 7.4)."]))
    temp_txt = _render_temperature(rng, temp_k)
    if "temperature_K" in missing:
        sentences.append("Measurement temperature per the instrument log (see metadata).")
    else:
        sentences.append(rng.choice([f"Equilibrated and measured at {temp_txt}.",
                                     f"Cell temperature held at {temp_txt} throughout the run."]))
    if "method" in missing:
        sentences.append(f"Run on the {instrument}; technique recorded in metadata.")
    else:
        sentences.append(rng.choice([f"Measured by {method} on the {instrument}.",
                                     f"{method} acquisition, {rng.randint(3, 12)} replicates, operator {operator}."]))
    sentences.append(rng.choice([f"Entry by {operator}, {date}.", f"Logged {date}.", ""]))
    rng.shuffle(sentences[1:4])  # vary order of the middle sentences
    text = " ".join(s for s in sentences if s)

    lookup = {
        "sample_id": f"Sample ID: {sample_id}",
        "temperature": f"Measurement temperature: {_render_temperature(rng, temp_k)}",
        "concentration": f"Sample concentration: {_render_concentration(rng, conc_mm)}",
        "method": f"Technique: {method}",
        "operator": f"Operator: {operator}",
        "date": f"Date: {date}",
        "instrument": f"Instrument: {instrument}",
        "buffer": f"Buffer: {buffer_conc} mM {buffer}, pH 7.4",
    }
    task = Task(
        task_id=f"toy-{idx:04d}",
        description=TASK_DESCRIPTION.format(n_max=n_max),
        context={
            "notebook_text": text,
            "lookup": lookup,
            "lookup_fields": LOOKUP_FIELDS,
            "n_max_tool_calls": n_max,
            "output_fields": FIELDS,
        },
    )
    return task, gt


def generate(seed: int, n_total: int, n_train: int, n_max: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    for i in range(n_total):
        task, gt = make_task(rng, i, n_max)
        records.append({"task": task.model_dump(), "ground_truth": gt, "split": "train" if i < n_train else "heldout"})
    return records


def main() -> None:
    import yaml

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=cfg["tasks"]["seed"])
    p.add_argument("--n-total", type=int, default=cfg["tasks"]["n_total"])
    p.add_argument("--n-train", type=int, default=cfg["tasks"]["n_train"])
    p.add_argument("--out", type=Path, default=root / "tasks" / "toy_v1.jsonl")
    args = p.parse_args()
    records = generate(args.seed, args.n_total, args.n_train, cfg["reward"]["n_max_tool_calls"])
    with args.out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} tasks -> {args.out}")


if __name__ == "__main__":
    main()
