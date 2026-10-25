"""make analyze RUN=<id>: reward / KL / rejection curves from experiments/<run>/metrics.jsonl.

With several runs (--compare), also writes the B0/B1/B2 table (Phase 3)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(run_dir: Path) -> tuple[list[dict], list[dict]]:
    train, held = [], []
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        (held if r.get("kind") == "heldout" else train).append(r)
    return train, held


def train_budget(train: list[dict]) -> tuple[int, int]:
    """(tokens, harness calls) spent on the G x tasks training rollouts only, excluding selection and held-out
    evaluation. Older runs logged only `*_total` (tokens: train-only; calls: train + held-out), so the training
    call count is rebuilt from per-step trajectory and rejection counts."""
    if not train:
        return 0, 0
    e = train[-1]["extra"]
    tokens = int(e.get("tokens_train", e["tokens_total"]))
    if "harness_calls_train" in e:
        return tokens, int(e["harness_calls_train"])
    calls = sum(round(r["extra"]["n_trajectories"] * (1 - r["extra"]["rejection_rate"])) for r in train)
    return tokens, calls


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
    tokens, calls = train_budget(train)
    return {"run": run_dir.name, "steps": len(train), "final_train": train[-1]["mean_reward"] if train else None,
            "heldout_b0": last_h.get("b0"), "heldout_best": last_h.get("best_bundle"), "heldout_policy": last_h.get("policy_sample"),
            "train_tokens": tokens, "train_harness_calls": calls,
            "all_harness_calls": train[-1]["extra"]["harness_calls_total"] if train else 0, "plot": str(path)}


def compare(spec: dict[str, dict[str, str]], exp: Path, out: Path) -> list[dict[str, Any]]:
    """spec: {domain: {"B1": run_id, "B2": run_id}} -> Phase 3 table (markdown + json) and bar figure.

    B0 = hand-written prompts (held-out, from any run's heldout record; cost 0).
    B1 = random-perturbation keep-best (held-out score of its best bundle; cost = its training rollouts).
    B2 = GRPO (held-out score of best bundle and of a fresh policy sample; cost = its training rollouts).
    Cost columns count training rollouts only (selection-slice and held-out evaluations excluded).
    """
    rows: list[dict[str, Any]] = []
    for domain, runs in spec.items():
        t1, h1 = load_metrics(exp / runs["B1"])
        t2, h2 = load_metrics(exp / runs["B2"])
        b0 = h2[-1]["b0"]
        tok1, calls1 = train_budget(t1)
        tok2, calls2 = train_budget(t2)
        rows.append({"domain": domain, "method": "B0 hand-written", "heldout": b0, "heldout_policy_sample": None,
                     "opt_tokens": 0, "opt_harness_calls": 0})
        rows.append({"domain": domain, "method": "B1 random keep-best", "heldout": h1[-1].get("best_bundle"), "heldout_policy_sample": h1[-1].get("policy_sample"),
                     "opt_tokens": tok1, "opt_harness_calls": calls1})
        rows.append({"domain": domain, "method": "B2 GRPO meta-policy", "heldout": h2[-1].get("best_bundle"), "heldout_policy_sample": h2[-1].get("policy_sample"),
                     "opt_tokens": tok2, "opt_harness_calls": calls2})
    out.mkdir(parents=True, exist_ok=True)
    md = ["| domain | method | held-out reward (best bundle) | held-out reward (policy sample) | training tokens | training harness calls |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        ps = "-" if r["heldout_policy_sample"] is None else f"{r['heldout_policy_sample']:.3f}"
        md.append(f"| {r['domain']} | {r['method']} | {r['heldout']:.3f} | {ps} | {r['opt_tokens']:,} | {r['opt_harness_calls']} |")
    (out / "baselines_table.md").write_text("\n".join(md) + "\n")
    (out / "baselines.json").write_text(json.dumps(rows, indent=2))
    domains = list(spec)
    methods = ["B0 hand-written", "B1 random keep-best", "B2 GRPO meta-policy"]
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.25
    for j, m in enumerate(methods):
        vals = [next(r["heldout"] for r in rows if r["domain"] == d and r["method"] == m) for d in domains]
        ax.bar([i + (j - 1) * w for i in range(len(domains))], vals, w, label=m)
        for i, v in enumerate(vals):
            ax.text(i + (j - 1) * w, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(domains))); ax.set_xticklabels(domains); ax.set_ylabel("held-out reward"); ax.legend(fontsize=8)
    ax.set_title("B0 / B1 / B2 under identical optimisation budgets"); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out / "baselines.png", dpi=130); plt.close(fig)
    print("\n".join(md))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="*", help="run ids under experiments/ to plot")
    p.add_argument("--experiments-dir", default="experiments")
    p.add_argument("--out", default=None)
    p.add_argument("--compare", default=None, help='JSON {"toy": {"B1": run, "B2": run}, ...} -> Phase 3 table + figure')
    a = p.parse_args()
    exp = Path(a.experiments_dir)
    out = Path(a.out) if a.out else exp / "analysis"
    rows = [plot_run(exp / r, out) for r in a.runs]
    for r in rows:
        print(json.dumps(r))
    if rows:
        (out / "summary.json").write_text(json.dumps(rows, indent=2))
    if a.compare:
        compare(json.loads(a.compare), exp, out)


if __name__ == "__main__":
    main()
