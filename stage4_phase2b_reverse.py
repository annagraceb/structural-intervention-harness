#!/usr/bin/env python3
"""
Stage 4 Phase 2B: Reverse-ordering cascade to test phase transition artifact.

The original cascade added heads sorted by ascending individual delta:
  [2, 7, 9, 1, 6, 0, 11, 10, 3]  (weakest first)

This reverses: strongest first, plus adds 7-head and 8-head trials.
If the "phase transition at 9 heads" is a real nonlinear effect, we should
see it regardless of ordering. If it's an artifact, the reverse ordering
will show a different pattern.
"""
import json
import os
import sys

import torch

import config
config.configure_determinism()

from benchmark import load_bbh_tasks, load_model_and_tokenizer
from weight_manager import WeightManager
from trial_runner import run_trial, print_trial_summary
from database import init_db, get_completed_trial_ids
from stage4_phase2 import multi_head_ablation


def main():
    print("=" * 60)
    print("PHASE 2B: Reverse-Order Cascade (strongest first)")
    print("=" * 60)

    items = load_bbh_tasks(config.BBH_TASKS)
    model, tokenizer = load_model_and_tokenizer(config.MODEL_ID, config.DEVICE)
    wm = WeightManager(model)
    db_conn = init_db()
    completed = get_completed_trial_ids(db_conn)

    baseline_path = os.path.join(config.BASELINE_DIR, "authoritative_baseline.json")
    with open(baseline_path) as f:
        baseline_results = json.load(f)

    # Reverse order: strongest individual ablation delta first
    # H3(+0.9), H10(+0.9), H11(+0.8), H0(+0.7), H6(+0.6), H1(+0.4),
    # H9(+0.3), H7(+0.3), H2(+0.0)
    reverse_order = [3, 10, 11, 0, 6, 1, 9, 7, 2]

    trials = []
    for n in [2, 3, 4, 6, 7, 8, 9]:
        heads = reverse_order[:n]
        trials.append({
            'trial_id': f"S4-REV-{n:02d}H",
            'category': 'S4-REV',
            'intervention_spec': {
                'type': 'multi_head_ablation',
                'layer': 9,
                'heads': heads,
                'description': f'L9 ablate {n} heads (REVERSE order): {heads}',
            },
        })

    pending = [t for t in trials if t['trial_id'] not in completed]
    print(f"  Trials: {len(trials)} total, {len(pending)} pending")

    # Verify weight restoration actually works now
    print(f"\n  Verify fix test: {wm.verify()}")
    sys.stdout.flush()

    for i, trial_spec in enumerate(pending):
        trial_id = trial_spec['trial_id']
        spec = trial_spec['intervention_spec']
        print(f"\n[{i+1}/{len(pending)}] {trial_id}: {spec['description']}")
        sys.stdout.flush()

        try:
            weight_mods = multi_head_ablation(wm, spec['layer'], spec['heads'])

            result = run_trial(
                model, tokenizer, items, baseline_results, wm,
                trial_id=trial_id,
                category=trial_spec['category'],
                intervention_spec=spec,
                weight_modifications=weight_mods,
                activation_hook=None,
                db_conn=db_conn,
                device=config.DEVICE,
            )
            print_trial_summary(result)
            sys.stdout.flush()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            wm.restore()

    # Analysis: compare forward vs reverse cascade
    print("\n" + "=" * 60)
    print("COMPARISON: Forward vs Reverse Cascade")
    print("=" * 60)

    print(f"\n  {'N':>3}  {'Forward (weak→strong)':>22}  {'Reverse (strong→weak)':>22}")
    print("  " + "-" * 52)

    for n in [2, 3, 4, 6, 7, 8, 9]:
        fwd_id = f"S4-MULTI-{n:02d}H"
        rev_id = f"S4-REV-{n:02d}H"

        fwd = db_conn.execute(
            'SELECT accuracy_delta FROM trials WHERE trial_id=?', (fwd_id,)
        ).fetchone()
        rev = db_conn.execute(
            'SELECT accuracy_delta FROM trials WHERE trial_id=?', (rev_id,)
        ).fetchone()

        fwd_str = f"{fwd[0]:+.1f}pp" if fwd else "N/A"
        rev_str = f"{rev[0]:+.1f}pp" if rev else "N/A"
        print(f"  {n:>3}  {fwd_str:>22}  {rev_str:>22}")

    print(f"\n  If phase transition is REAL: both should jump at ~9 heads")
    print(f"  If ordering ARTIFACT: reverse should jump at ~2-3 heads")

    db_conn.close()
    print("\nPhase 2B complete.")


if __name__ == "__main__":
    main()
