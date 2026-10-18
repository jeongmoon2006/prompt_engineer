# Phase 2 / Phase 3 report — GRPO meta-policy and baselines

Date: 2026-10-18. Harness model: `claude-haiku-5-5` (effort low, thinking disabled). Meta-policy: `Qwen/Qwen2.5-0.5B-Instruct` + LoRA (r=16, alpha=32, q/k/v/o), trained on a GTX 1660 Super (6 GB).
All numbers below come from `experiments/<run>/metrics.jsonl`; figures are in `experiments/analysis/`.

## 1. What was built (Phase 2)

| Component | File | Notes |
|---|---|---|
| Meta-policy pi_phi | `orchestrator/policy.py` | Fixed task-independent meta-prompt built from `/domain_info`; emits `<system>…</system><planner>…</planner><tool_policy>…</tool_policy>`; pi_ref = same weights with the adapter disabled |
| GRPO | `orchestrator/grpo.py` | Group-standardised advantages, clipped surrogate (eps 0.2), per-token k3 KL to pi_ref (beta 0.04), one AdamW step per batch, GradScaler fp16 autocast, grad-clip 1.0 |
| Trainer | `orchestrator/trainer.py` | Batched sampling of all G x tasks completions, rollout via harness `/run_batch` + env `/evaluate`, held-out eval every 2 steps, LoRA checkpoints every 5, best bundle written as drop-in YAML |
| Degeneracy guards | `orchestrator/policy.py::parse_bundle` | Missing/duplicate/stray tags, empty slot, > 1500 chars per slot, extra text outside tags, non-UTF8 -> reward -1 |
| Analysis | `orchestrator/analyze.py` | Reward / KL / rejection curves, B0/B1/B2 table and figure |

Budget per step: 8 tasks x G=8 = 64 trajectories. 20 steps. Held-out: 16 tasks, fixed seed.

## 2. Phase 2 result — 20 GRPO steps on the toy domain (`toy-grpo`)

Figure: `experiments/analysis/toy-grpo_curves.png`

| step | train mean R (all 64) | train mean R (valid only) | rejection | KL(pi_phi ‖ pi_ref) | mean bundle length (chars) |
|---|---|---|---|---|---|
| 0 | -0.93 | 0.46 | 0.95 | 0.000 | 742 |
| 5 | -0.39 | 0.50 | 0.59 | 0.033 | 685 |
| 10 | -0.04 | 0.54 | 0.38 | 0.125 | 364 |
| 15 | 0.21 | 0.55 | 0.22 | 0.271 | 277 |
| 19 | 0.37 | 0.54 | 0.11 | 0.329 | 207 |

Held-out (16 tasks):

| | reward | correctness | format | tool_eff | cost |
|---|---|---|---|---|---|
| B0 hand-written prompts | 0.555 | 1.00 | 0.75 | 0.47 | -0.59 |
| best GRPO bundle (train R 0.651, found at step 17) | 0.575 | 0.94 | 0.88 | 0.47 | -0.54 |
| policy sample after step 20 | 0.554 | 1.00 | 0.75 | 0.47 | -0.59 |

Checkpoint-2 criterion ("held-out should rise above the baseline prompt's score"): the best bundle does (0.575 vs 0.555), a sample from the final policy ties it (0.554). The train curve rises from -0.93 to +0.37, but almost all of that is the rejection rate falling from 0.95 to 0.11; the mean reward of *valid* bundles moves only 0.46 -> 0.54 and never clearly exceeds B0.

### Diagnosis (as the checkpoint asks)

- **Reward variance within groups.** Within-group std of the 8 rewards is 0.4-0.7 while rejections are frequent, but the std among *valid* bundles in the same group is only 0.02-0.13 (median ~0.06). So > 90 % of the advantage signal is "emit three well-formed tags"; the residual signal among valid prompts is small and comes mostly from the cost term.
- **What the residual signal selects for.** Haiku 5.5 solves the toy task from the task text and tool schema alone (correctness 0.95-1.0 regardless of prompt wording), so among valid bundles the cheapest prompt wins. Mean bundle length fell 742 -> 207 chars and late-stage prompts are short near-nonsense ("Read your current text without any items, measurements, labels, or spaces, please."). The reward is doing exactly what it says; the harness is simply insensitive to prompt content on this domain.
- **Rejection rate.** 0.95 -> 0.11; monotone apart from noise. Remaining rejections are mostly `tag_count_*=0` and `extra_text`.
- **KL.** No collapse: KL per token grows smoothly 0 -> 0.33 with beta 0.04; ratio stays 1 by construction (one update per batch), so clipping never triggers. Grad norm 0.3 -> 1.0.
- **Temperature.** Sampling at tau = 1.0 (the meta-policy's own temperature; the frozen Haiku harness takes none). At step 0 this produced 95 % malformed output from the 0.5B base model; lowering tau would cut early rejections but also the exploration GRPO needs. Not changed.
- **frac_zero_adv_groups** was 0.0 in every step, so no group was wasted.

Conclusion for Phase 2: GRPO on a 0.5B LoRA meta-policy works mechanically (stable training, no KL collapse, 4.1 GB peak VRAM, ~7 min/step). The ceiling is the reward landscape, not the optimiser: on this harness + domain the prompt bundle explains little of the reward variance.

## 3. Phase 3 — baselines B0 / B1 / B2, two domains

Same budget for B1 and B2: 20 steps x 64 harness trajectories, held-out 16 tasks, identical seeds. B0 is re-measured inside each run; across runs it varies by ±0.005 (toy 0.555 / 0.560, gsm8k 0.670 / 0.671), which is the Haiku-side noise floor on 16 tasks. B1 = random keep-best (synonym swaps and sentence shuffles of the hand-written bundle, G=8 per step; keep the best by train reward). B2 = GRPO. "Optimisation tokens" = Haiku input+output tokens spent during the 20 training steps (held-out evals excluded). GRPO makes fewer harness calls than B1 because rejected bundles never reach the harness.

Figure: `experiments/analysis/baselines.png`; table: `experiments/analysis/baselines_table.md`.

| domain | method | held-out reward (best bundle) | held-out reward (policy sample) | optimisation tokens | harness calls |
|---|---|---|---|---|---|
| toy | B0 hand-written | 0.555 | - | 0 | 0 |
| toy | B1 random keep-best | 0.562 | 0.561 | 3,175,914 | 1600 |
| toy | B2 GRPO meta-policy | 0.575 | 0.554 | 1,824,642 | 1004 |
| gsm8k | B0 hand-written | 0.670 | - | 0 | 0 |
| gsm8k | B1 random keep-best | 0.686 | 0.675 | 1,203,990 | 1600 |
| gsm8k | B2 GRPO meta-policy | 0.656 | 0.100 (valid-only 0.601; 31 % rejected) | 1,118,682 | 1001 |

Second domain: GSM8K (`openai/gsm8k`, main split; 300 train / 100 held-out frozen in `env/tasks/gsm8k_v1.jsonl`) with a `calculator` tool (safe AST evaluator, N_max 6), exact numeric match for correctness. Unlike the toy domain, Haiku's correctness is *not* fully saturated here: train-batch correctness ranges 0.75-1.0 (mean 0.95) versus 0.98-1.0 on toy, so the correctness term carries some signal; the 16 held-out tasks happen to be solved by every bundle tried.

GRPO on GSM8K (`gsm8k-grpo`, figure `experiments/analysis/gsm8k-grpo_curves.png`): train mean reward -0.90 -> +0.37, rejection 0.94 -> 0.16, KL 0 -> 0.15 (one spike to 0.24 at step 14, recovered). Held-out: the best bundle scores 0.656, *below* B0 (0.671). It had scored 0.772 on its 8-task training batch at step 12 and did not transfer: keep-best on 8 tasks is a noisy selector (the toy best bundle likewise shrank from 0.651 train to 0.575 held-out). A sample from the final policy scores 0.100 because 5 of 16 samples were still malformed; the 11 valid ones average 0.601. The mean reward of valid bundles stayed flat at 0.52-0.65 through the whole run versus 0.67 for B0, so the meta-policy never produced prompts as good as the hand-written ones. Bundle length fell 663 -> 326 chars; within-group std among valid bundles 0.02-0.12, the same picture as on toy.

### Reading the table

- On the toy domain all three methods land within 0.02 of each other, which is inside the noise of a 16-task held-out set (one task flipping fenced/bare JSON moves the mean by 0.019). Neither optimiser beats B0 convincingly, and B1 spends 1.7x the tokens of B2 for the same outcome.
- On GSM8K the ordering is B1 > B0 > B2. B1 only paraphrases the hand-written bundle, so it cannot drift far from B0; GRPO explores freely from a weak 0.5B prior and, within this budget, never catches up. GRPO is also 2x cheaper per step in harness calls because rejected bundles are scored locally, but that saving buys nothing when the surviving prompts are worse.
- Held-out sets of 16 tasks and keep-best selection on 8-task batches are both too small: train-batch bests of 0.651 / 0.772 became 0.575 / 0.656 held-out. Phase 4 should evaluate candidate bundles on >= 32 held-out tasks before declaring a winner.

## 4. Deviations from WORK_INSTRUCTIONS.md (all deliberate, none silent)

| Item | Spec | What was done | Why |
|---|---|---|---|
| Reward weights | 0.6 / 0.15 / 0.15 / 0.10 | 0.4 correctness / 0.3 format / 0.1 tool / 0.2 cost | Haiku saturates correctness on the toy domain; approved by user before Phase 2 |
| Learning rate | 1e-5 | 1e-4 | LoRA with only 20 optimizer steps; 1e-5 was judged too small to move the policy measurably (recorded in `orchestrator/config.yaml`) |
| Harness thinking | not specified | disabled | keeps per-trajectory cost and latency bounded; Haiku 5.5 rejects non-default temperature anyway |
| Extra endpoints | `/health`, `/run_batch`, `/evaluate`, `/step` | + `/tasks`, `/domain_info` (env), `/prompts` (harness) | orchestrator must fetch tasks and baseline prompts over HTTP; nothing shares a filesystem |
| Docker | `make smoke` via compose | compose files written, unverified; `make smoke-local` / `make serve-local` run the three services as host processes | no Docker in this WSL2 install |
| Meta-prompt | per-domain | one generic template filled from `/domain_info` | needed so B2 could be re-run on GSM8K without touching the orchestrator |

## 5. Lessons

1. **Measure where the reward variance lives before optimising prompts.** With a strong frozen model, prompt wording explained almost none of the reward on the toy task; GRPO then correctly optimised the only thing left (format validity, then length). Pick domains and reward terms where B0 is clearly below ceiling.
2. **Rejection dominates early GRPO on a small meta-policy.** A 0.5B model at tau = 1.0 is 95 % malformed at step 0. Group standardisation handles the -1 rewards fine, but the first ~10 steps are spent on syntax, not semantics. A format-constrained decoder or a short SFT warm-up on valid bundles would buy those steps back.
3. **Cost terms create a length-collapse attractor.** Any reward with -tokens/budget and a prompt-insensitive harness converges to the shortest parseable prompt. Cap the cost term or make it relative to B0's cost rather than absolute.
4. **Compare against random search at equal budget.** B1 (paraphrase + keep-best) matched GRPO on toy and beat it on GSM8K while being far simpler and never rejecting a sample; that is the honest headline for the mentor and the motivation for Phase 4 (DSPy / GEPA, which start from the hand-written prompt instead of a 0.5B prior) and Phase 5 (conditioning).
5. **Engineering on a 6 GB card is fine.** fp32 master + fp16 autocast, gradient checkpointing, selective log-softmax and batched generation keep a 0.5B LoRA GRPO step at ~4 GB and ~7 min; generation, not the update, is the bottleneck on WSL.

## 6. How to reproduce

```bash
make serve-local                      # harness :8001, env :8002 (host processes)
make train RUN=toy-grpo               # 20 GRPO steps on toy (~2.5 h on the 1660 Super)
make train RUN=toy-random  POLICY=random
make train RUN=gsm8k-grpo  DOMAIN=gsm8k
make train RUN=gsm8k-random POLICY=random DOMAIN=gsm8k
make analyze RUN=toy-grpo RUNS=gsm8k-grpo \
  COMPARE='{"toy": {"B1": "toy-random", "B2": "toy-grpo"}, "gsm8k": {"B1": "gsm8k-random", "B2": "gsm8k-grpo"}}'
```
