# Pre-Registered Predictions

**Frozen:** 2026-03-25T08:30Z (after 27/612 W1 trials complete; no significant results observed)
**Model:** Qwen2.5-1.5B (fp16, SDPA)
**Benchmark:** 1000 BBH items, 29.0% baseline

> **Timing note:** These predictions were written after Stage 3 execution began but
> before any non-W1 results were available. Only W1 (permutation) trials had run,
> producing no significant effects (all p > 0.05). Predictions below are grounded
> in transformer architecture theory, not observed experimental data.

---

## Predictions

### Weight interventions

**P1. MLP reinitializations are more destructive than attention reinitializations.**
Reinitializing (W3-kaiming) any single MLP block in layers 4-24 will decrease accuracy
by at least 5pp. Reinitializing any single attention block (all of Q/K/V/O) in the
same layer range will decrease accuracy by less than 5pp on average.
*Rationale:* MLPs store factual/procedural knowledge; attention routes information
but is more redundant due to multi-head parallelism.

**P2. Middle layers are more critical than early/late layers.**
For W3 (reinitialization) and A3 (layer skip), applying the intervention to layers
in the middle third (layers 9-18) will produce larger |accuracy_delta| than applying
it to layers in the first third (0-8) or last third (19-27).
*Rationale:* Middle layers perform the core reasoning computation; early layers
handle token embedding and late layers handle output formatting.

**P3. Adjacent-layer transplants are less destructive than distant transplants.**
W2 transplants between adjacent layers (|src-tgt| ≤ 2) will produce |accuracy_delta|
< 2pp on average. Transplants between distant layers (|src-tgt| ≥ 10) will produce
|accuracy_delta| > 3pp on average.
*Rationale:* Adjacent layers learn similar representations due to residual stream
continuity; distant layers are more specialized.

**P4. Head ablation in the last 3 layers has minimal impact.**
Ablating (W4-ablate, zeroing o_proj slice) any single attention head in layers 25-27
will change accuracy by at most 2pp.
*Rationale:* Late-layer heads primarily handle output token selection, which for
log-prob evaluation is less critical than mid-layer reasoning heads.

**P5. Spectral truncation preserving top 50% of singular values is benign.**
W5 top-k truncation with k = rank/2 on any Q or V projection will preserve at
least 90% of baseline accuracy (i.e., accuracy ≥ 26.1%).
*Rationale:* Weight matrices in trained models have rapidly decaying singular value
spectra; the bottom half contributes little to the output.

**P6. Permutations (W1) are the weakest intervention class.**
No W1 trial will achieve Tier 1 significance. The average |accuracy_delta| across
all W1 trials will be less than 2pp.
*Rationale:* Permutations preserve matrix norms, eigenvalue distributions, and
information content — they only rearrange internal indexing.

### Activation interventions

**P7. Full layer skip (A3) of any single layer degrades accuracy.**
Skipping any single layer (A3-full) will decrease accuracy by at least 1pp.
No single-layer skip will improve accuracy.
*Rationale:* Each layer adds information to the residual stream; skipping one
removes that contribution, which should hurt performance.

**P8. Attention-only skip is less destructive than full layer skip.**
For any given layer, A3-attention_only will produce smaller |accuracy_delta| than
A3-full for the same layer.
*Rationale:* The MLP sublayer contributes more to the residual stream magnitude
than the attention sublayer, so skipping attention alone removes less information.

**P9. Scaling attention heads to zero (A2, scalar=0) is equivalent to ablation (W4).**
For any given head, the accuracy under A2 scalar=0 will be within 0.5pp of
W4-ablate on the same head.
*Rationale:* Both interventions zero out the head's contribution to the output.
Minor differences may arise from numerical precision of hook vs weight zeroing.

**P10. Residual stream noise injection (A1) is monotonically destructive.**
Increasing the A1 injection magnitude (alpha) will monotonically decrease accuracy.
No A1 trial with alpha > 0 will improve accuracy over baseline.
*Rationale:* Adding noise to the residual stream should only corrupt the signal.

### Cross-cutting predictions

**P11. Five-object tasks are more sensitive to interventions than three-object tasks.**
For any Tier 1 intervention, the accuracy delta on 5-object items will have larger
magnitude than on 3-object items.
*Rationale:* Harder tasks (5-object, lower baseline) require more precise internal
computation, making them more fragile to perturbation.

**P12. Tracking tasks are more sensitive than deduction tasks.**
For any Tier 1 intervention, the accuracy delta on tracking tasks will have larger
magnitude than on deduction tasks.
*Rationale:* Tracking requires maintaining state across sequential operations;
deduction requires constraint satisfaction. Sequential state tracking relies more
heavily on precise layer-by-layer computation.

**P13. No intervention will improve accuracy by more than 5pp.**
The maximum accuracy_delta across all 612 trials will be less than +5.0pp.
*Rationale:* The model is near-chance (29%); structural damage should degrade
performance, and improvement would require the intervention to somehow fix a
systematic error — unlikely from blind structural modifications.

**P14. Degenerate trials will be concentrated in W3 and W5.**
At least 80% of degenerate trials (NaN/uniform logprobs) will come from W3
(reinitialization) or W5 (spectral editing) categories.
*Rationale:* These interventions most aggressively modify weight magnitudes,
which can push activations into numerical overflow/underflow in fp16.

---

## Evaluation protocol

Each prediction is evaluated as TRUE or FALSE after the coarse sweep completes.
A Tier 3 result requires:
1. The result meets Tier 2 criteria (BH-significant, |delta| ≥ 3pp, ≥ 70% directional)
2. The result violates at least one prediction above
3. The specific violated prediction is cited in the anomaly report
