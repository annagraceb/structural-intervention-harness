#!/usr/bin/env python3
"""
Stage 4 Phase 1: GPU experiments for theory discrimination.
1A: Complete L9 per-head ablation/negate sweep (16 trials)
1B: Combined intervention tests (5 trials)
1C: Cross-task generalization (3 trials)
"""
import json
import os
import sys
import time

import torch

import config
config.configure_determinism()

from benchmark import load_bbh_tasks, load_model_and_tokenizer, run_benchmark, evaluate_item_logprob
from weight_manager import WeightManager
from interventions import (
    w4_head_surgery, w3_reinitialize, w5_spectral_edit,
    a2_head_scaling_hook, a3_layer_skip_hook, ActivationHook,
)
from trial_runner import run_trial, print_trial_summary
from database import init_db, get_completed_trial_ids, save_trial
from stage3_run import execute_intervention


def generate_phase1_trials(wm):
    """Generate all Phase 1 trial specs."""
    trials = []

    # === 1A: Complete L9 per-head sweep ===
    # Ablate: heads 1, 2, 4, 5, 6, 8, 9, 10 (0, 3, 7, 11 already done in coarse sweep)
    tested_heads = {0, 3, 7, 11}
    for head in range(12):
        if head not in tested_heads:
            trials.append({
                'trial_id': f"S4-L9-ABL-H{head:02d}",
                'category': 'S4-L9',
                'intervention_spec': {
                    'type': 'W4_head_surgery',
                    'layer': 9,
                    'operation': 'ablate',
                    'head_i': head,
                },
            })
            trials.append({
                'trial_id': f"S4-L9-NEG-H{head:02d}",
                'category': 'S4-L9',
                'intervention_spec': {
                    'type': 'W4_head_surgery',
                    'layer': 9,
                    'operation': 'negate',
                    'head_i': head,
                },
            })

    # === 1B: Combined intervention tests ===
    # C1: L9 attn zero + L18 H11 ablate
    trials.append({
        'trial_id': "S4-COMB-01",
        'category': 'S4-COMB',
        'intervention_spec': {
            'type': 'combined',
            'components': [
                {'type': 'W3_reinitialize', 'layer': 9, 'granularity': 'attention', 'distribution': 'zero', 'seed': 3020},
                {'type': 'W4_head_surgery', 'layer': 18, 'operation': 'ablate', 'head_i': 11},
            ],
            'description': 'L9 attn zero + L18 H11 ablate',
        },
    })
    # C2: L9 attn zero + L20 gate uniform spectrum
    trials.append({
        'trial_id': "S4-COMB-02",
        'category': 'S4-COMB',
        'intervention_spec': {
            'type': 'combined',
            'components': [
                {'type': 'W3_reinitialize', 'layer': 9, 'granularity': 'attention', 'distribution': 'zero', 'seed': 3020},
                {'type': 'W5_spectral_edit', 'layer': 20, 'component': 'mlp.gate_proj.weight', 'operation': 'uniform_spectrum', 'k': None},
            ],
            'description': 'L9 attn zero + L20 gate uniform spectrum',
        },
    })
    # C3: All three weight-space improvements
    trials.append({
        'trial_id': "S4-COMB-03",
        'category': 'S4-COMB',
        'intervention_spec': {
            'type': 'combined',
            'components': [
                {'type': 'W3_reinitialize', 'layer': 9, 'granularity': 'attention', 'distribution': 'zero', 'seed': 3020},
                {'type': 'W4_head_surgery', 'layer': 18, 'operation': 'ablate', 'head_i': 11},
                {'type': 'W5_spectral_edit', 'layer': 20, 'component': 'mlp.gate_proj.weight', 'operation': 'uniform_spectrum', 'k': None},
            ],
            'description': 'L9 attn zero + L18 H11 ablate + L20 gate uniform',
        },
    })
    # C4: L9 attn skip + L6 H5 scale 5x (hook-based combo)
    trials.append({
        'trial_id': "S4-COMB-04",
        'category': 'S4-COMB',
        'intervention_spec': {
            'type': 'combined_hooks',
            'components': [
                {'type': 'A3_layer_skip', 'layer': 9, 'variant': 'attention_only'},
                {'type': 'A2_head_scaling', 'layer': 6, 'head_idx': 5, 'scalar': 5.0},
            ],
            'description': 'L9 attn skip + L6 H5 scale 5x',
        },
    })
    # C5: All five distinct improvements
    trials.append({
        'trial_id': "S4-COMB-05",
        'category': 'S4-COMB',
        'intervention_spec': {
            'type': 'combined_all',
            'components': [
                {'type': 'W3_reinitialize', 'layer': 9, 'granularity': 'attention', 'distribution': 'zero', 'seed': 3020},
                {'type': 'W4_head_surgery', 'layer': 18, 'operation': 'ablate', 'head_i': 11},
                {'type': 'W5_spectral_edit', 'layer': 20, 'component': 'mlp.gate_proj.weight', 'operation': 'uniform_spectrum', 'k': None},
                {'type': 'A2_head_scaling', 'layer': 6, 'head_idx': 5, 'scalar': 5.0},
                {'type': 'A3_layer_skip', 'layer': 25, 'variant': 'full'},
            ],
            'description': 'All 5 improvements combined',
        },
    })

    return trials


def execute_combined_intervention(model, wm, spec, device):
    """Execute a combined intervention (multiple specs merged)."""
    all_weight_mods = {}
    all_hooks = []

    for component_spec in spec['components']:
        if component_spec['type'].startswith('W'):
            mods, _ = execute_intervention(model, wm, component_spec, device)
            if mods:
                all_weight_mods.update(mods)
        elif component_spec['type'].startswith('A'):
            _, hook = execute_intervention(model, wm, component_spec, device)
            if hook:
                all_hooks.append(hook)

    # Merge hooks into a single ActivationHook-like object
    combined_hook = None
    if all_hooks:
        combined_hook = ActivationHook()
        for h in all_hooks:
            combined_hook._handles.extend(h._handles)

    return all_weight_mods or None, combined_hook


def run_cross_task_trials(model, tokenizer, wm, db_conn, device):
    """Experiment 1C: Test L9 attention zero on other BBH tasks."""
    print("\n--- Experiment 1C: Cross-Task Generalization ---")

    extra_tasks = ["boolean_expressions", "navigate", "web_of_lies"]

    for task_name in extra_tasks:
        trial_id = f"S4-XGEN-{task_name[:8]}"

        # Check if already done
        completed = get_completed_trial_ids(db_conn)
        if trial_id + "-baseline" in completed and trial_id + "-interv" in completed:
            print(f"  {task_name}: already done, skipping")
            continue

        print(f"\n  Loading task: {task_name}")
        try:
            items = load_bbh_tasks([task_name])
        except Exception as e:
            print(f"  Failed to load {task_name}: {e}")
            continue

        print(f"  {len(items)} items loaded")

        # Run baseline
        print(f"  Running baseline...")
        baseline_results = []
        for item in items:
            r = evaluate_item_logprob(model, tokenizer, item, device)
            r['item_id'] = item['item_id']
            r['task'] = item['task']
            baseline_results.append(r)

        bl_correct = sum(1 for r in baseline_results if r['correct'])
        bl_acc = bl_correct / len(baseline_results) * 100
        print(f"  Baseline: {bl_correct}/{len(baseline_results)} = {bl_acc:.1f}%")

        # Apply L9 attention zero
        spec = {'type': 'W3_reinitialize', 'layer': 9, 'granularity': 'attention', 'distribution': 'zero', 'seed': 3020}
        mods, _ = execute_intervention(model, wm, spec, device)

        # Run intervention
        print(f"  Running L9 attn zero intervention...")
        wm.apply_weight_modification(mods)

        int_results = []
        for item in items:
            r = evaluate_item_logprob(model, tokenizer, item, device)
            r['item_id'] = item['item_id']
            r['task'] = item['task']
            int_results.append(r)

        # Restore weights
        wm.restore()
        if not wm.verify():
            print(f"  CRITICAL: Weight verification failed!")

        int_correct = sum(1 for r in int_results if r['correct'])
        int_acc = int_correct / len(int_results) * 100
        delta = int_acc - bl_acc

        # McNemar
        flipped_correct = 0
        flipped_incorrect = 0
        for bl_r, int_r in zip(baseline_results, int_results):
            if bl_r['correct'] and not int_r['correct']:
                flipped_incorrect += 1
            elif not bl_r['correct'] and int_r['correct']:
                flipped_correct += 1

        from statistics import mcnemar_exact_test
        p_val = mcnemar_exact_test(b=flipped_incorrect, c=flipped_correct)

        print(f"  {task_name}: baseline={bl_acc:.1f}%, intervention={int_acc:.1f}%, "
              f"delta={delta:+.1f}pp, +{flipped_correct}/-{flipped_incorrect}, p={p_val:.4f}")


def main():
    print("=" * 60)
    print("STAGE 4 PHASE 1: GPU Experiments")
    print("=" * 60)

    # Load everything
    items = load_bbh_tasks(config.BBH_TASKS)
    model, tokenizer = load_model_and_tokenizer(config.MODEL_ID, config.DEVICE)
    wm = WeightManager(model)
    db_conn = init_db()
    completed = get_completed_trial_ids(db_conn)

    # Load baseline
    baseline_path = os.path.join(config.BASELINE_DIR, "authoritative_baseline.json")
    with open(baseline_path) as f:
        baseline_results = json.load(f)

    # Generate trials
    all_trials = generate_phase1_trials(wm)
    pending = [t for t in all_trials if t['trial_id'] not in completed]
    print(f"\n  Total Phase 1 trials: {len(all_trials)}, pending: {len(pending)}")

    start_time = time.time()

    # Run trials
    for i, trial_spec in enumerate(pending):
        trial_id = trial_spec['trial_id']
        spec = trial_spec['intervention_spec']

        print(f"\n[{i+1}/{len(pending)}] {trial_id}: {spec.get('description', spec.get('type', ''))}")

        try:
            if spec['type'].startswith('combined'):
                weight_mods, hook = execute_combined_intervention(model, wm, spec, config.DEVICE)
            else:
                weight_mods, hook = execute_intervention(model, wm, spec, config.DEVICE)

            result = run_trial(
                model, tokenizer, items, baseline_results, wm,
                trial_id=trial_id,
                category=trial_spec['category'],
                intervention_spec=spec,
                weight_modifications=weight_mods,
                activation_hook=hook,
                db_conn=db_conn,
                device=config.DEVICE,
            )
            print_trial_summary(result)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            wm.restore()

    # Run cross-task generalization
    run_cross_task_trials(model, tokenizer, wm, db_conn, config.DEVICE)

    total_time = (time.time() - start_time) / 60
    print(f"\n  Phase 1 complete in {total_time:.1f} minutes")

    # Analyze L9 head results
    print("\n" + "=" * 60)
    print("ANALYSIS: L9 Per-Head Profile")
    print("=" * 60)

    # Combine coarse sweep + new S4 results
    rows = db_conn.execute("""
        SELECT trial_id, intervention_spec, accuracy_delta, mcnemar_p_value
        FROM trials
        WHERE is_degenerate=0
        AND (
            (category='W4' AND json_extract(intervention_spec, '$.layer')=9)
            OR category='S4-L9'
        )
        ORDER BY trial_id
    """).fetchall()

    ablate_by_head = {}
    negate_by_head = {}
    for r in rows:
        spec = json.loads(r[1])
        head = spec.get('head_i', spec.get('head_idx'))
        if head is None:
            continue
        if spec['operation'] == 'ablate':
            ablate_by_head[head] = (r[2], r[3])
        elif spec['operation'] == 'negate':
            negate_by_head[head] = (r[2], r[3])

    print(f"\n  {'Head':>4}  {'Ablate Δ':>10}  {'Ablate p':>10}  {'Negate Δ':>10}  {'Negate p':>10}")
    print("  " + "-" * 50)
    for h in range(12):
        abl = ablate_by_head.get(h, (None, None))
        neg = negate_by_head.get(h, (None, None))
        a_str = f"{abl[0]:+.1f}pp" if abl[0] is not None else "   N/A"
        ap_str = f"{abl[1]:.4f}" if abl[1] is not None else "  N/A"
        n_str = f"{neg[0]:+.1f}pp" if neg[0] is not None else "   N/A"
        np_str = f"{neg[1]:.4f}" if neg[1] is not None else "  N/A"
        print(f"  {h:>4}  {a_str:>10}  {ap_str:>10}  {n_str:>10}  {np_str:>10}")

    # Sum of individual ablations
    abl_sum = sum(d for d, p in ablate_by_head.values() if d is not None)
    print(f"\n  Sum of individual ablation deltas: {abl_sum:+.1f}pp")
    print(f"  Full L9 attention zero (W3-0020): +3.7pp")
    if abl_sum > 2.5:
        print(f"  -> Effect is distributed (sum ≈ full zero)")
    else:
        print(f"  -> Effect may be synergistic (sum < full zero)")

    # Combined intervention results
    print("\n" + "=" * 60)
    print("ANALYSIS: Combined Interventions")
    print("=" * 60)

    comb_rows = db_conn.execute("""
        SELECT trial_id, intervention_spec, accuracy_delta, mcnemar_p_value,
               items_flipped_to_correct, items_flipped_to_incorrect
        FROM trials WHERE category='S4-COMB' AND is_degenerate=0
        ORDER BY trial_id
    """).fetchall()

    for r in comb_rows:
        spec = json.loads(r[1])
        desc = spec.get('description', r[0])
        print(f"  {r[0]}: {desc}")
        print(f"    Δ={r[2]:+.1f}pp  p={r[3]:.4f}  +{r[4]}/-{r[5]}")

    if comb_rows:
        max_single = 3.7  # W3-0020
        max_combined = max(r[2] for r in comb_rows)
        all5 = [r for r in comb_rows if 'All 5' in json.loads(r[1]).get('description', '')]
        if all5:
            all5_delta = all5[0][2]
            print(f"\n  Max single intervention: +{max_single:.1f}pp")
            print(f"  All-5 combined: {all5_delta:+.1f}pp")
            if all5_delta > max_single + 2:
                print(f"  -> T8 SUPPORTED: Improvements COMPOUND ({all5_delta:+.1f} > {max_single:+.1f})")
            elif all5_delta >= max_single - 1:
                print(f"  -> T5 SUPPORTED: Improvements PLATEAU (shared bottleneck)")
            else:
                print(f"  -> SURPRISING: Combined is WORSE than individual")

    db_conn.close()


if __name__ == "__main__":
    main()
