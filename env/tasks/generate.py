"""Toy domain task generator: synthetic lab-notebook extraction.

Deterministic given the seed. Writes env/tasks/<file>.jsonl with one record per line:
{"task": Task, "ground_truth": {...}, "split": "train"|"heldout"}.

The agent sees only `Task` (description + context). Ground truth stays in this
file and is read by the environment, never sent to the harness.

Difficulty "hard" (v2, default) adds: dilution arithmetic (final = stock / factor, or
volumes), corrected values that supersede earlier ones, incubation / storage / set-point
temperatures that are not the measurement temperature, methods spelled out in full,
reference samples, and decoy lookup fields. "easy" reproduces v1.

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
METHODS = {
    "DLS": "dynamic light scattering", "NMR": "nuclear magnetic resonance", "UV-Vis": "UV-visible spectroscopy",
    "SAXS": "small-angle X-ray scattering", "ITC": "isothermal titration calorimetry",
    "cryo-EM": "cryo-electron microscopy", "FTIR": "Fourier-transform infrared spectroscopy",
    "HPLC": "high-performance liquid chromatography",
}
BUFFERS = ["Tris", "HEPES", "PBS", "MES", "phosphate"]
OPERATORS = ["J.K.", "M.R.", "A.S.", "L.W.", "T.N."]
INSTRUMENTS = ["Zetasizer Nano", "Bruker 600", "Cary 60", "Xeuss 3.0", "MicroCal PEAQ", "Titan Krios"]
LOOKUP_FIELDS_EASY = ["sample_id", "temperature", "concentration", "method", "operator", "date", "instrument", "buffer"]
LOOKUP_FIELDS_HARD = LOOKUP_FIELDS_EASY + ["storage_temperature", "stock_concentration", "reference_sample", "incubation", "molecular_weight"]

TASK_DESCRIPTION = (
    "Extract a structured record from the lab notebook entry. Fields: sample_id (string, the sample "
    "that was measured); temperature_K (number, the temperature at which the measurement itself was "
    "made, in kelvin); concentration_mM (number, the final concentration of the measured sample in "
    "millimolar); method (standard abbreviation of the measurement technique). Some values are not "
    "written in the entry and must be retrieved with the lookup tool, which may be called at most "
    "{n_max} times. Answer with one JSON object."
)


def _num(x: float) -> str:
    return f"{x:g}"


def _temp_txt(rng: random.Random, k: float) -> str:
    return f"{_num(round(k - 273.15, 2))} °C" if rng.random() < 0.55 else f"{_num(k)} K"


def _conc_txt(rng: random.Random, mm: float) -> str:
    r = rng.random()
    if r < 0.3 and mm < 1.0:
        return f"{_num(round(mm * 1000, 4))} µM"
    if r < 0.45 and mm >= 100:
        return f"{_num(round(mm / 1000, 5))} M"
    return f"{_num(mm)} mM"


MW_KDA = [5, 10, 12.5, 20, 25, 50]


def _mass_conc_txt(rng: random.Random, mm: float, mw_kda: float) -> str:
    """mM x kDa == mg/mL exactly (mmol/L x g/mmol)."""
    mg_ml = round(mm * mw_kda, 5)
    if mg_ml < 0.1 and rng.random() < 0.5:
        return f"{_num(round(mg_ml * 1000, 4))} µg/mL (MW {_num(mw_kda)} kDa)"
    return f"{_num(mg_ml)} mg/mL (MW {_num(mw_kda)} kDa)"


def _sample_id(rng: random.Random) -> str:
    return rng.choice(["S-", "LN-", "BX", "P"]) + f"{rng.randint(1, 9999):04d}" + rng.choice(["", "A", "B", "-r2"])


def make_task(rng: random.Random, idx: int, n_max: int, hard: bool) -> tuple[Task, dict[str, Any]]:
    sample_id = _sample_id(rng)
    ref_id = _sample_id(rng)
    temp_k = round(rng.choice([277.15, 283.15, 293.15, 295.15, 298.15, 303.15, 310.15, 318.15, 323.15, 333.15])
                   + rng.choice([0.0, 0.0, 0.5, -0.5, 1.0, 2.0]), 2)
    conc_mm = round(rng.choice([0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 150.0, 250.0])
                    * rng.choice([1.0, 1.0, 1.2, 0.8]), 4)
    method = rng.choice(list(METHODS))
    gt = {"sample_id": sample_id, "temperature_K": temp_k, "concentration_mM": conc_mm, "method": method}

    buffer_conc, buffer = rng.choice([10, 20, 25, 50, 100]), rng.choice(BUFFERS)
    storage_k = rng.choice([253.15, 277.15, 193.15])
    incub_k = rng.choice([310.15, 298.15, 333.15])
    setpoint_k = round(temp_k - rng.choice([1.5, 2.3, 3.0, -2.0]), 2)
    operator, instrument = rng.choice(OPERATORS), rng.choice(INSTRUMENTS)
    date = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    other_method = rng.choice([m for m in METHODS if m != method])
    factor = rng.choice([2, 4, 5, 10, 20, 50])
    stock_mm = round(conc_mm * factor, 4)
    v1 = rng.choice([10, 20, 25, 50, 100])
    vtot = v1 * factor
    mw_kda = rng.choice(MW_KDA)

    n_missing = rng.choice([1, 1, 2])
    missing = set(rng.sample(FIELDS, n_missing))
    S: list[str] = []

    # --- sample id
    if "sample_id" in missing:
        S.append(rng.choice(["Sample ID recorded in the instrument metadata.",
                             f"Reported sample as per metadata; {ref_id} and others were run alongside." if hard else "Sample ID recorded in the instrument metadata."]))
    elif hard and rng.random() < 0.5:
        S.append(rng.choice([f"Measured {sample_id}; {ref_id} served as the reference sample (not reported here).",
                             f"Thawed {sample_id} and control {ref_id} from {_num(storage_k)} K storage; only {sample_id} was measured."]))
    else:
        S.append(rng.choice([f"Sample {sample_id} was taken from the freezer stock (stored at {_num(storage_k)} K).",
                             f"Worked on {sample_id} today; aliquot thawed from {_num(storage_k)} K storage."]))

    # --- concentration
    if "concentration_mM" in missing:
        S.append(f"Diluted into {buffer_conc} mM {buffer} buffer, pH 7.4; final sample concentration logged in metadata.")
    elif hard:
        v = rng.random()
        if v < 0.2:
            S.append(f"Stock at {_conc_txt(rng, stock_mm)} diluted 1:{factor} into {buffer_conc} mM {buffer}, pH 7.4.")
        elif v < 0.35:
            S.append(f"{v1} µL of {_conc_txt(rng, stock_mm)} stock brought to {vtot} µL with {buffer_conc} mM {buffer} (pH 7.4).")
        elif v < 0.5:
            wrong = round(conc_mm * rng.choice([1.25, 0.8, 2.0]), 4)
            S.append(f"Diluted to {_conc_txt(rng, wrong)} in {buffer_conc} mM {buffer} — correction after re-weighing: {_conc_txt(rng, conc_mm)}.")
        elif v < 0.7 and conc_mm * mw_kda <= 50:  # keep mass concentrations physically plausible (<= 50 mg/mL)
            S.append(f"Working concentration {_mass_conc_txt(rng, conc_mm, mw_kda)} in {buffer_conc} mM {buffer}, pH 7.4.")
        elif v < 0.9:
            # Multi-sample run: the reported sample's value is one of several.
            others = [(ref_id, round(conc_mm * rng.choice([2.0, 0.5, 4.0]), 4)), (_sample_id(rng), round(conc_mm * rng.choice([1.5, 0.25]), 4))]
            entries = [(sample_id, conc_mm)] + others
            rng.shuffle(entries)
            S.append("Run as a series in " + f"{buffer_conc} mM {buffer}: " + "; ".join(f"{sid} at {_conc_txt(rng, c)}" for sid, c in entries) + ".")
        else:
            S.append(f"Final concentration {_conc_txt(rng, conc_mm)} ({buffer_conc} mM {buffer}, pH 7.4; stock was {_conc_txt(rng, stock_mm)}).")
    else:
        S.append(rng.choice([f"Diluted to {_conc_txt(rng, conc_mm)} in {buffer_conc} mM {buffer} buffer, pH 7.4.",
                             f"Final concentration {_conc_txt(rng, conc_mm)} ({buffer_conc} mM {buffer}, pH 7.4)."]))

    # --- temperature
    if "temperature_K" in missing:
        S.append(rng.choice(["Measurement temperature per the instrument log (see metadata).",
                             f"Pre-incubated at {_temp_txt(rng, incub_k)}; measurement temperature in metadata." if hard else "Measurement temperature per the instrument log (see metadata)."]))
    elif hard:
        v = rng.random()
        if v < 0.35:
            S.append(f"Pre-incubated {rng.randint(10, 60)} min at {_temp_txt(rng, incub_k)}, then measured at {_temp_txt(rng, temp_k)}.")
        elif v < 0.65:
            S.append(f"Set point {_temp_txt(rng, setpoint_k)}; cell thermocouple read {_temp_txt(rng, temp_k)} during acquisition — use the cell reading.")
        else:
            S.append(f"Measured at {_temp_txt(rng, temp_k)} (bath nominally {_temp_txt(rng, setpoint_k)}; samples otherwise kept at {_temp_txt(rng, storage_k)}).")
    else:
        S.append(rng.choice([f"Equilibrated and measured at {_temp_txt(rng, temp_k)}.",
                             f"Cell temperature held at {_temp_txt(rng, temp_k)} throughout the run."]))

    # --- method
    if "method" in missing:
        S.append(f"Run on the {instrument}; technique recorded in metadata." if not hard or rng.random() < 0.5
                 else f"Earlier batches were characterized by {other_method}; today's technique is recorded in metadata.")
    elif hard:
        v = rng.random()
        if v < 0.4:
            S.append(f"Characterized by {METHODS[method]} on the {instrument}.")
        elif v < 0.7:
            S.append(f"Previously screened by {other_method}; this entry: {METHODS[method]} ({method}) acquisition, {rng.randint(3, 12)} replicates.")
        else:
            S.append(f"Measured by {method} on the {instrument} (the {other_method} run is scheduled for next week).")
    else:
        S.append(rng.choice([f"Measured by {method} on the {instrument}.",
                             f"{method} acquisition, {rng.randint(3, 12)} replicates, operator {operator}."]))

    S.append(rng.choice([f"Entry by {operator}, {date}.", f"Logged {date}.", ""]))
    mid = S[1:4]
    rng.shuffle(mid)
    S[1:4] = mid
    text = " ".join(s for s in S if s)

    lookup = {
        "sample_id": f"Sample ID: {sample_id}",
        "temperature": f"Measurement temperature: {_temp_txt(rng, temp_k)}",
        "concentration": (f"Sample concentration: stock {_conc_txt(rng, stock_mm)} diluted 1:{factor}" if hard and (lv := rng.random()) < 0.3
                          else f"Sample concentration: {_mass_conc_txt(rng, conc_mm, mw_kda)}" if hard and lv < 0.55 and conc_mm * mw_kda <= 50
                          else f"Sample concentration: {_conc_txt(rng, conc_mm)}"),
        "method": f"Technique: {METHODS[method]}" if hard and rng.random() < 0.5 else f"Technique: {method}",
        "operator": f"Operator: {operator}",
        "date": f"Date: {date}",
        "instrument": f"Instrument: {instrument}",
        "buffer": f"Buffer: {buffer_conc} mM {buffer}, pH 7.4",
    }
    if hard:
        lookup.update({
            "storage_temperature": f"Storage temperature: {_num(storage_k)} K",
            "stock_concentration": f"Stock concentration: {_conc_txt(rng, stock_mm)}",
            "reference_sample": f"Reference sample: {ref_id}",
            "incubation": f"Pre-incubation: {_temp_txt(rng, incub_k)}",
            "molecular_weight": f"Molecular weight: {_num(mw_kda)} kDa",
        })
    task = Task(
        task_id=f"toy-{idx:04d}",
        description=TASK_DESCRIPTION.format(n_max=n_max),
        context={
            "notebook_text": text,
            "lookup": lookup,
            "lookup_fields": LOOKUP_FIELDS_HARD if hard else LOOKUP_FIELDS_EASY,
            "n_max_tool_calls": n_max,
            "output_fields": FIELDS,
        },
    )
    return task, gt


def generate(seed: int, n_total: int, n_train: int, n_max: int, difficulty: str = "hard") -> list[dict[str, Any]]:
    rng = random.Random(seed)
    hard = difficulty == "hard"
    records = []
    for i in range(n_total):
        task, gt = make_task(rng, i, n_max, hard)
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
    p.add_argument("--difficulty", choices=["easy", "hard"], default=cfg["tasks"].get("difficulty", "hard"))
    p.add_argument("--out", type=Path, default=root / "tasks" / cfg["tasks"].get("file", "toy_v2.jsonl"))
    args = p.parse_args()
    records = generate(args.seed, args.n_total, args.n_train, cfg["reward"]["n_max_tool_calls"], args.difficulty)
    with args.out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} {args.difficulty} tasks -> {args.out}")


if __name__ == "__main__":
    main()
