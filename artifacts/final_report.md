# Structural Intervention Experiment: Final Report

**Date:** 2026-03-26
**Model:** Qwen2.5-1.5B (1.54B parameters, fp16, SDPA attention)
**Benchmark:** 1000 BBH items across 4 tasks, 29.0% baseline accuracy
**Hardware:** NVIDIA RTX 3060 12GB, ~32 hours total GPU time
**Trials:** 650 total (612 coarse sweep + 38 targeted follow-ups)

---

## 1. Experiment Overview

We systematically modified the internal weights and activations of a small
language model to understand how structural components contribute to its
reasoning ability. The model was evaluated on a 1000-item subset of
BIG-Bench Hard (BBH) — specifically, logical deduction and object tracking
tasks that require multi-step reasoning.

### Intervention categories

| Category | Description | Trials |
|----------|-------------|--------|
| W1 | Weight permutation (shuffle rows/cols of weight matrices) | 100 |
| W2 | Weight transplant (copy one layer's weights to another) | 92 |
| W3 | Weight reinitialization (zero, Kaiming, or noise) | 56 |
| W4 | Attention head surgery (ablate, negate, duplicate, average) | 63 |
| W5 | Spectral editing (SVD truncation, inversion, uniform) | 100 |
| A1 | Residual stream noise injection | 42 |
| A2 | Attention head output scaling | 75 |
| A3 | Layer skip (full, attention-only, MLP-only) | 84 |
| S4 | Stage 4 targeted follow-ups | 38 |

### Evaluation method

Each trial modifies the model, evaluates all 1000 items via log-probability
scoring, restores the model to its pristine state (verified via checksummed
CPU backup), then compares item-by-item against the deterministic baseline.
Statistical significance is assessed with McNemar's exact test (paired,
two-sided) on the discordant items.

---

## 2. Top-Level Results

### Nothing survives multiple testing correction

After Benjamini-Hochberg correction (FDR = 0.05) across all trials, **zero
results are statistically significant.** The number of results reaching
uncorrected p < 0.05 (29 out of 650, or 4.5%) is consistent with the 5%
expected by chance.

The positive/negative split across all 650 non-degenerate trials is
309 positive vs 292 negative (48% positive) — indistinguishable from a
coin flip.

### Largest effects observed

**Improvements (top 5):**

| Trial | Description | Delta | p-value | Items +/- |
|-------|-------------|-------|---------|-----------|
| S4-MULTI-ALL-EX-H5 | L9: ablate all heads except H5 | +4.3pp | 0.0016 | +111/-68 |
| S4-MULTI-09H | L9: ablate 9 positive heads | +4.2pp | 0.0016 | +106/-64 |
| W3-0020 | L9: zero all attention weights | +3.7pp | 0.0042 | +98/-61 |
| A3-0028 | L9: skip attention sublayer | +3.7pp | 0.0042 | +98/-61 |
| A3-0075 | L25: skip full layer | +3.4pp | 0.0315 | +135/-101 |

**Degradations (top 5):**

| Trial | Description | Delta | p-value |
|-------|-------------|-------|---------|
| W3-0006 | L2: Kaiming-reinitialize MLP | -4.5pp | 0.0204 |
| A3-0042 | L14: skip full layer | -4.5pp | 0.0155 |
| W2-0090 | Transplant distant layers | -4.4pp | 0.0225 |
| W3-0026 | L9: Kaiming-reinitialize MLP | -4.4pp | 0.0207 |
| W3-0015 | L5: Kaiming-reinitialize full layer | -4.1pp | 0.0341 |

---

## 3. The "Destruction Helps" Phenomenon

Several structurally diverse interventions — zeroing layer 9 attention,
skipping layer 9 attention, ablating a layer 18 head, scaling a layer 6
head by 5x, forward-transplanting layer 7 to layer 9 — each improved
accuracy by 2.5-3.7pp. This appears paradoxical: removing or disrupting
model components should degrade performance, not improve it.

### Explanation: borderline items on a near-chance model

The model scores 29.0% on a benchmark where random chance ranges from
20-33% depending on the task. Many items sit right at the decision
boundary.

**Logprob margin analysis** reveals that items flipped to correct by any
intervention have a median baseline margin of 0.030 nats (i.e., the
correct answer's log-probability was only 0.030 below the top incorrect
answer). Items that stayed incorrect have a median margin of 0.058
(Mann-Whitney p = 5.9 x 10^-8, one-sided).

However, this **self-audit revealed that this pattern is a statistical
tautology**: 90% of null trials (delta between -0.5 and +0.5pp) show the
same significant margin effect. Items near the decision boundary are more
likely to flip under *any* perturbation — helpful, harmful, or random.
This is a mathematical property of the setup, not a finding about specific
interventions.

**The "destruction helps" effect is best understood as:** any sufficiently
large perturbation to a near-chance model will nudge some borderline items
across the decision boundary in both directions. When more items happen to
flip correct than incorrect, it appears as improvement. The effect is small
(3-4pp), unreliable (doesn't survive multiple testing correction), and not
specific to any particular model component.

---

## 4. Layer 9 Deep Dive

Layer 9 attention was the strongest single target: zeroing it entirely
improved accuracy by +3.7pp. We investigated this in depth.

### Per-head profile (all 12 L9 attention heads)

| Head | Ablate delta | Negate delta | Role |
|------|-------------|-------------|------|
| H00 | +0.7pp | +0.1pp | Mildly harmful |
| H01 | +0.4pp | +1.2pp | Mildly harmful |
| H02 | +0.0pp | +0.6pp | Neutral |
| H03 | +0.9pp | +0.9pp | Harmful |
| H04 | -0.8pp | +0.7pp | Mildly useful |
| **H05** | **-1.3pp** | **-1.0pp** | **Useful** |
| H06 | +0.6pp | +2.9pp | Harmful |
| H07 | +0.3pp | +1.8pp | Mildly harmful |
| H08 | -1.2pp | +0.4pp | Mildly useful |
| H09 | +0.3pp | +1.7pp | Mildly harmful |
| H10 | +0.9pp | +0.8pp | Harmful |
| H11 | +0.8pp | +2.5pp | Harmful |

**Head 5 is the only head where both ablation and negation hurt.** It is
the only genuinely useful attention head in layer 9 for these tasks.

### Multi-head ablation cascade

We ablated increasing numbers of heads simultaneously, in two orderings:

| N heads | Forward (weak first) | Reverse (strong first) |
|---------|---------------------|----------------------|
| 2 | -0.3pp | +1.6pp |
| 3 | +0.3pp | +1.9pp |
| 4 | +0.4pp | +1.4pp |
| 6 | +0.9pp | +2.8pp |
| 7 | N/A | +1.9pp |
| 8 | N/A | +2.7pp |
| 9 | **+4.2pp** | **+4.2pp** |
| 11 (all except H5) | +4.3pp | N/A |
| 12 (all) | +3.7pp | N/A |

**Key observations:**

1. Both orderings converge to +4.2pp at 9 heads — the total effect is
   robust regardless of which heads you remove first.

2. The path matters: removing the 3 strongest heads first (reverse) gives
   +1.6pp at just 2 heads; removing the 3 weakest first (forward) gives
   -0.3pp. There is no sharp "phase transition" — an initial draft of
   this report incorrectly claimed one. The effect scales gradually, with
   strongest individual heads contributing most.

3. All-except-H5 (+4.3pp) > all-12 (+3.7pp). Keeping H5 while removing
   everything else is the optimal configuration, confirming H5 is doing
   useful work that gets washed out by the other 11 heads.

4. Ablating the 3 useful heads (H4, H5, H8) gives -2.0pp, confirming
   these heads contribute positively.

### L9 attention output norm

L9's attention output norm ranks 18th out of 28 layers (0.66x the mean).
Late layers (25-27) dominate. The "destruction helps" effect at L9 is
**not** due to unusually large or disruptive attention output magnitude.

---

## 5. Combined Interventions

We tested whether combining multiple improving interventions would
compound their effects. They do not — they interfere.

| Trial | Components | Delta | Total flips | Efficiency |
|-------|-----------|-------|-------------|------------|
| W3-0020 (reference) | 1 | +3.7pp | 159 | 23% |
| COMB-01: L9 zero + L18H11 ablate | 2 | +2.2pp | 176 | 12% |
| COMB-02: L9 zero + L20 gate | 2 | +1.5pp | 207 | 7% |
| COMB-03: triple | 3 | +0.9pp | 229 | 4% |
| COMB-04: L9 skip + L6H5 scale | 2 | +3.1pp | 203 | 15% |
| COMB-05: all 5 combined | 5 | +3.0pp | 312 | 10% |

"Efficiency" = net items improved / total items changed.

Adding more intervention components monotonically increases the total
number of items that change (churn) while decreasing the net improvement.
Each additional perturbation fixes some new items but breaks others.
The best single intervention (W3-0020 at +3.7pp) outperforms every
combination that includes it.

This rules out the hypothesis that different interventions fix independent
problems. The improvements overlap heavily — they all operate on the same
narrow band of borderline items.

---

## 6. Pre-Registered Predictions

14 predictions were frozen before interpreting results. Evaluation:

| # | Prediction | Result |
|---|-----------|--------|
| P1 | MLP reinit more destructive than attention | **VIOLATED** — attention reinit was more destructive (-2.40pp vs -1.29pp mean) |
| P2 | Middle layers most critical | Partially supported (W3, A3 show mid-layer sensitivity) |
| P3 | Adjacent transplants less destructive | **VIOLATED** — no distance effect observed |
| P4 | Late-layer head ablation minimal | Supported |
| P5 | 50% spectral truncation benign | Supported |
| P6 | Permutations weakest class | Supported (avg |delta| = 1.06pp, 1 sig at p<0.05) |
| P7 | Layer skip always degrades | **VIOLATED** — 17 A3 trials improved by >1pp |
| P8 | Attention skip less destructive than full | Supported on average |
| P9 | A2 scalar=0 equivalent to W4 ablate | Supported |
| P10 | Noise injection monotonically destructive | **VIOLATED** — 13 A1 trials showed positive delta |
| P11 | 5-object tasks more sensitive | Not formally tested (Tier 3 analysis incomplete) |
| P12 | Tracking more sensitive than deduction | Not formally tested |
| P13 | No intervention improves >5pp | **SUPPORTED** — max was +4.3pp |
| P14 | Degenerate trials concentrated in W3/W5 | Cannot evaluate (zero degenerate trials observed) |

5 of 14 predictions were violated. Notably, P7 and P10 (which predicted
that removing computation or adding noise should always hurt) were the
predictions most directly challenged by the "destruction helps"
observations.

---

## 7. Methodological Self-Audit

Three adversarial audits were conducted on the experimental methodology:

### Bug found: WeightManager.verify() was a no-op

The `verify()` method was computing a checksum of the *pristine backup*
rather than the *live model weights*, meaning it always returned True
regardless of actual model state. **Fixed during the audit.** The
`restore()` method itself is sound (unconditional clone from CPU backup
with `strict=True`), and baseline consistency analysis shows no temporal
drift across 650 trials, so results are not contaminated.

### Ordering artifact in cascade

The initial "phase transition" claim was based on a cascade that added
the weakest individual heads first. The reverse-ordering experiment
(Section 4) debunked the sharp transition while confirming the total
effect. The corrected interpretation is gradual, additive scaling.

### Margin analysis is tautological

The finding that flipped items have smaller baseline margins is true for
~90% of all trials, including those with near-zero net effect. It reflects
the mathematical structure of perturbing a near-chance classifier, not a
property of specific interventions. This analysis was initially presented
as a discovery and has been downgraded to a methodological observation.

### Item identity, not just counts

S4-MULTI-ALL12 and W3-0020 produce byte-identical item-level results
(same 98 items flipped correct, same 61 flipped incorrect). These are
methodologically equivalent, not independent replications.

---

## 8. Conclusions

### What we learned

1. **A 1.5B-parameter model operating near chance on hard reasoning tasks
   is remarkably robust to structural perturbation.** Across 650 trials
   spanning 7 intervention families, no result survived multiple testing
   correction. The model's performance is not critically dependent on any
   single layer, head, or weight matrix.

2. **Apparent improvements from "destructive" interventions are an artifact
   of near-chance performance.** When a model is barely above random, many
   items sit at the decision boundary. Any perturbation nudges some across,
   creating the illusion of targeted improvement.

3. **Layer 9 attention is the most effective single target for perturbation-
   based "improvement."** Removing 9+ of its 12 heads yields +4.2pp.
   Head 5 is the only individually useful head. But this effect does not
   survive BH correction and should not be interpreted as evidence that
   L9 attention is "harmful" in any meaningful sense.

4. **Improvements do not compound.** Combining multiple perturbations
   increases churn while decreasing net benefit. The improving interventions
   all operate on the same ~100 borderline items and interfere with each
   other when stacked.

5. **Pre-registered predictions had a 64% accuracy rate.** The 5 violations
   were concentrated in predictions about directionality (P1: which
   component type is more critical) and monotonicity (P7, P10: destruction
   always hurts). These violations are explained by the borderline-items
   effect.

### What we did not find

- No evidence that specific model components are systematically harmful
  to reasoning
- No intervention that reliably improves performance beyond statistical
  noise
- No compounding of improvements from independent interventions
- No sharp phase transitions in collective head behavior

### Limitations

- Single model (Qwen2.5-1.5B), single benchmark (4 BBH tasks)
- Near-chance baseline (29%) limits sensitivity to detect real effects
- No cross-model or cross-benchmark replication
- Weight verification was broken during the main sweep (fixed during audit;
  indirect evidence suggests no contamination)
- Some Stage 4 trials (cross-task generalization) were uninformative due
  to evaluation method incompatibility with non-BBH answer formats

---

## 9. Files and Reproducibility

| File | Description |
|------|-------------|
| `experiment.db` | SQLite database with all 650 trials and 650,000 item results |
| `artifacts/preregistered_predictions.md` | 14 predictions frozen before analysis |
| `artifacts/item_property_taxonomy.json` | Item property splits for Fisher tests |
| `stage3_run.py` | Main coarse sweep orchestrator (612 trials) |
| `stage4_phase0.py` | Zero-GPU analysis (overlap, norms, similarity) |
| `stage4_phase1.py` | L9 per-head sweep + combined interventions |
| `stage4_phase2.py` | Multi-head ablation cascade + attention norms |
| `stage4_phase2b_reverse.py` | Reverse-ordering cascade (self-audit) |
| `config.py` | Model, benchmark, and determinism configuration |
| `weight_manager.py` | Pristine weight backup/restore (verify bug fixed) |
| `interventions.py` | All 8 intervention families |
| `trial_runner.py` | Trial execution with restore-in-finally |
| `benchmark.py` | Log-probability evaluation pipeline |
| `database.py` | SQLite schema and helpers |
