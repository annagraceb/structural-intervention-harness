# Structural Intervention Experiment — Anomaly Report

**Generated:** /home/cisco/structural_intervention_harness/experiment.db
**Total trials:** 612 non-degenerate
**Tier 2+ results:** 0

## Pre-Registered Prediction Evaluation

**Violated:** 5 | **Confirmed:** 5 | **Inconclusive:** 0

!!! **P1** [VIOLATED]: MLP reinit ≥5pp degradation; attention reinit <5pp avg
    Evidence: MLP reinit avg Δ=-1.3pp (all≤-5pp: False), Attention reinit avg |Δ|=2.1pp

!!! **P2_W3** [VIOLATED]: Middle layers more critical for W3
    Evidence: Middle |Δ| avg=1.7pp, Edge |Δ| avg=1.8pp

 ?  **P2_A3** [INCONCLUSIVE]: Middle layers more critical for A3
    Evidence: Middle |Δ| avg=1.9pp, Edge |Δ| avg=1.3pp

!!! **P3** [VIOLATED]: Adjacent transplants <2pp, distant >3pp
    Evidence: Adjacent avg |Δ|=1.4pp, Distant avg |Δ|=0.9pp

    **P4** [CONFIRMED]: Head ablation in last 3 layers ≤2pp
    Evidence: Deltas: [0.1, -0.3, 0.1, 0.4]

    **P5** [CONFIRMED]: Spectral top-50% preserves ≥90% accuracy
    Evidence: Threshold=0.261, accuracies=[0.291, 0.29, 0.303, 0.305, 0.293]

 ?  **P6** [INCONCLUSIVE]: No W1 reaches Tier 1; avg |Δ| < 2pp
    Evidence: BH-significant W1: False, avg |Δ|=1.1pp

!!! **P7** [VIOLATED]: Full layer skip always degrades (≥1pp), never improves
    Evidence: Deltas: min=-4.5pp, max=+3.4pp, any_improve=True

    **P8** [CONFIRMED]: Attention-only skip less destructive than full skip
    Evidence: 10/28 layers violated (attn≥full)

!!! **P10** [VIOLATED]: No A1 injection improves accuracy
    Evidence: Max Δ=+1.3pp

    **P13** [CONFIRMED]: Max improvement < 5pp
    Evidence: Max accuracy_delta = +3.7pp

    **P14** [CONFIRMED]: ≥80% degenerate from W3/W5
    Evidence: No degenerate trials


## Tier 3 Candidates

No Tier 3 candidates found.


## Tier 2+ Results with Item-Property Analysis


## Category Summary

**W1:** 100 trials, 0 significant, 0 Tier 2+, avg Δ=-0.3pp, max |Δ|=3.5pp
**W2:** 92 trials, 0 significant, 0 Tier 2+, avg Δ=-0.4pp, max |Δ|=4.4pp
**W3:** 56 trials, 0 significant, 0 Tier 2+, avg Δ=-1.1pp, max |Δ|=4.5pp
**W4:** 63 trials, 0 significant, 0 Tier 2+, avg Δ=+0.1pp, max |Δ|=3.7pp
**W5:** 100 trials, 0 significant, 0 Tier 2+, avg Δ=+0.1pp, max |Δ|=2.8pp
**A1:** 42 trials, 0 significant, 0 Tier 2+, avg Δ=+0.1pp, max |Δ|=1.3pp
**A2:** 75 trials, 0 significant, 0 Tier 2+, avg Δ=+0.1pp, max |Δ|=3.2pp
**A3:** 84 trials, 0 significant, 0 Tier 2+, avg Δ=-0.6pp, max |Δ|=4.5pp


## Null Results (no significance despite reasonable power)

Categories with zero BH-significant results: W1, W2, W3, W4, W5, A1, A2, A3
These null results are informative — they suggest the model is robust to these intervention types.
