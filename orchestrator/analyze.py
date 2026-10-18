"""make analyze RUN=<id>: reward / KL / rejection curves from experiments/<run>/metrics.jsonl.

With several runs (--compare), also writes the B0/B1/B2 table (Phase 3)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_metrics(run_dir: Path) -> tuple[list[dict], list[dict]]:
    train, held = [], []
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        (held if r.get("kind") == "heldout" else train).append(r)
    return train, held


def plot_run(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    train, held = load_metrics(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = [r["step"] for r in train]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.plot(steps, [r["mean_reward"] for r in train], "o-", label="train mean (incl. rejected)")
    ax.plot(steps, [r["extra"]["mean_reward_valid"] for r in train], "s--", label="train mean (valid)")
    if held:
        hs = [h["step"] for h in held]
        ax.plot(hs, [h["b0"] for h in held], "k:", label="held-out B0 baseline")
        ax.plot(hs, [h.get("best_bundle") for h in held], "^-", label="held-out best bundle")
        ax.plot(hs, [h.get("policy_sample") for h in held], "v-", label="held-out policy sample")
    ax.set_xlabel("step"); ax.set_ylabel("reward"); ax.set_title(f"{run_dir.name}: reward"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax = axes[1]
    ax.plot(steps, [r["kl"] for r in train], "o-", color="C3")
    ax.set_xlabel("step"); ax.set_ylabel("KL(pi_phi || pi_ref) per token"); ax.set_title("KL"); ax.grid(alpha=.3)
    ax = axes[2]
    ax.plot(steps, [r["extra"]["rejection_rate"] for r in train], "o-", color="C2", label="rejection rate")
    ax.plot(steps, [r["extra"]["components_mean"].get("format", float("nan")) for r in train], "s-", label="format (valid)")
    ax.plot(steps, [-r["extra"]["components_mean"].get("cost", float("nan")) for r in train], "d-", label="-cost (valid)")
    ax.set_xlabel("step"); ax.set_title("rejection / format / cost"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout()
    path = out_dir / f"{run_dir.name}_curves.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    last_h = held[-1] if held else {}
    return {"run": run_dir.name, "steps": len(train), "final_train": train[-1]["mean_reward"] if train else None,
            "heldout_b0": last_h.get("b0"), "heldout_best": last_h.get("best_bundle"), "heldout_policy": last_h.get("policy_sample"),
            "tokens_total": train[-1]["extra"]["tokens_total"] if train else 0,
            "harness_calls": train[-1]["extra"]["harness_calls_total"] if train else 0, "plot": str(path)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run ids under experiments/")
    p.add_argument("--experiments-dir", default="experiments")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    exp = Path(a.experiments_dir)
    out = Path(a.out) if a.out else exp / "analysis"
    rows = [plot_run(exp / r, out) for r in a.runs]
    for r in rows:
        print(json.dumps(r))
    (out / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
