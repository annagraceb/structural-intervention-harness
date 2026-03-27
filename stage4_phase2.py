#!/usr/bin/env python3
"""
Stage 4 Phase 2: Theory discrimination experiments.

Key question: Why does zeroing ALL of L9 attention (+3.7pp) exceed the sum of
individual head ablations (+1.6pp)?

Experiments:
  2A: Multi-head ablation cascade (ablate N heads simultaneously)
  2B: Ablate all-except-H5 (keep only the useful head)
  2C: L9 attention output norm measurement (is L9 attention output unusually large?)
  2D: Repeat top interventions on shuffled item order (robustness check)
"""
import json
import os
import time
import itertools

import torch
import numpy as np

import config
config.configure_determinism()

from benchmark import load_bbh_tasks, load_model_and_tokenizer, evaluate_item_logprob
from weight_manager import WeightManager
from interventions import w4_head_surgery, ActivationHook
from trial_runner import run_trial, print_trial_summary
from database import init_db, get_completed_trial_ids
from stage3_run import execute_intervention


def multi_head_ablation(wm, layer_idx, heads_to_ablate):
    """Ablate multiple heads simultaneously by zeroing their o_proj columns."""
    params = wm.get_layer_param_names(layer_idx)
    head_dim = wm.get_head_dim()

    o_proj_name = None
    for name, full_name in params.items():
        if 'o_proj' in name and 'weight' in name:
            o_proj_name = full_name
            break
    if o_proj_name is None:
        raise ValueError(f"o_proj not found in layer {layer_idx}")

    o_weight = wm.get_param(o_proj_name).clone()
    for head_i in heads_to_ablate:
        start = head_i * head_dim
        end = (head_i + 1) * head_dim
        o_weight[:, start:end] = 0.0

    return {o_proj_name: o_weight}


def measure_l9_attention_norms(model, tokenizer, items, device, n_items=50):
    """Measure the L2 norm of attention output at each layer for a sample of items."""
    norms_by_layer = {l: [] for l in range(28)}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is (attn_output, attn_weights) or just attn_output
            if isinstance(output, tuple):
                attn_out = output[0]
            else:
                attn_out = output
            # Mean norm across sequence positions
            norm = attn_out.float().norm(dim=-1).mean().item()
            norms_by_layer[layer_idx].append(norm)
        return hook_fn

    # Register hooks on all attention layers
    for layer_idx in range(28):
        attn_module = model.model.layers[layer_idx].self_attn
        handle = attn_module.register_forward_hook(make_hook(layer_idx))
        hooks.append(handle)

    # Run on a sample of items
    sample = items[:n_items]
    for item in sample:
        evaluate_item_logprob(model, tokenizer, item, device)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute mean norms
    mean_norms = {}
    for l in range(28):
        mean_norms[l] = np.mean(norms_by_layer[l]) if norms_by_layer[l] else 0.0

    return mean_norms


def generate_phase2_trials():
    """Generate Phase 2 trial specs."""
    trials = []

    # === 2A: Multi-head ablation cascade ===
    # Ablate increasing numbers of heads, excluding H5 (useful) and H4/H8 (mildly useful)
    # Sort by ablation delta (most harmful first): H4(-0.8), H8(-1.2), H5(-1.3) are negative
    # Positive heads sorted by delta: H2(0.0), H7(0.3), H9(0.3), H1(0.4), H6(0.6), H0(0.7), H11(0.8), H10(0.9), H3(0.9)
    positive_heads = [2, 7, 9, 1, 6, 0, 11, 10, 3]  # sorted by ascending ablation delta

    for n in [2, 3, 4, 6, 9]:  # ablate this many positive heads
        heads = positive_heads[:n]
        trials.append({
            'trial_id': f"S4-MULTI-{n:02d}H",
            'category': 'S4-MULTI',
            'intervention_spec': {
                'type': 'multi_head_ablation',
                'layer': 9,
                'heads': heads,
                'description': f'L9 ablate {n} positive heads: {heads}',
            },
        })

    # === 2B: Ablate all except H5 ===
    all_except_5 = [h for h in range(12) if h != 5]
    trials.append({
        'trial_id': "S4-MULTI-ALL-EX-H5",
        'category': 'S4-MULTI',
        'intervention_spec': {
            'type': 'multi_head_ablation',
            'layer': 9,
            'heads': all_except_5,
            'description': 'L9 ablate all 11 heads except H5',
        },
    })

    # Ablate all 12 (equivalent to zeroing all attention output via o_proj)
    trials.append({
        'trial_id': "S4-MULTI-ALL12",
        'category': 'S4-MULTI',
        'intervention_spec': {
            'type': 'multi_head_ablation',
            'layer': 9,
            'heads': list(range(12)),
            'description': 'L9 ablate all 12 heads via o_proj',
        },
    })

    # === 2B-extra: Ablate only the 3 negative heads ===
    negative_heads = [4, 5, 8]  # the ones where ablation hurts
    trials.append({
        'trial_id': "S4-MULTI-NEG3",
        'category': 'S4-MULTI',
        'intervention_spec': {
            'type': 'multi_head_ablation',
            'layer': 9,
            'heads': negative_heads,
            'description': 'L9 ablate 3 negative heads (H4, H5, H8)',
        },
    })

    return trials


def main():
    print("=" * 60)
    print("STAGE 4 PHASE 2: Theory Discrimination")
    print("=" * 60)

    items = load_bbh_tasks(config.BBH_TASKS)
    model, tokenizer = load_model_and_tokenizer(config.MODEL_ID, config.DEVICE)
    wm = WeightManager(model)
    db_conn = init_db()
    completed = get_completed_trial_ids(db_conn)

    baseline_path = os.path.join(config.BASELINE_DIR, "authoritative_baseline.json")
    with open(baseline_path) as f:
        baseline_results = json.load(f)

    # === Experiment 2C: Attention output norms ===
    print("\n--- Experiment 2C: Attention Output Norms (50 items) ---")
    norms = measure_l9_attention_norms(model, tokenizer, items, config.DEVICE, n_items=50)
    norm_values = list(norms.values())
    mean_norm = np.mean(norm_values)
    l9_norm = norms[9]
    l9_rank = sorted(range(28), key=lambda l: norm_values[l], reverse=True).index(9) + 1
    print(f"  L9 attention output norm: {l9_norm:.2f} (rank {l9_rank}/28, mean={mean_norm:.2f})")
    print(f"  L9 / mean ratio: {l9_norm/mean_norm:.2f}x")

    # Print all norms sorted
    print(f"\n  {'Layer':>5}  {'Norm':>8}  {'Ratio':>8}")
    for l in sorted(range(28), key=lambda l: norm_values[l], reverse=True):
        marker = " <-- L9" if l == 9 else ""
        print(f"  {l:>5}  {norm_values[l]:>8.2f}  {norm_values[l]/mean_norm:>7.2f}x{marker}")

    # === Experiments 2A/2B: Multi-head ablation cascade ===
    print("\n--- Experiments 2A/2B: Multi-Head Ablation Cascade ---")
    all_trials = generate_phase2_trials()
    pending = [t for t in all_trials if t['trial_id'] not in completed]
    print(f"  Trials: {len(all_trials)} total, {len(pending)} pending")

    for i, trial_spec in enumerate(pending):
        trial_id = trial_spec['trial_id']
        spec = trial_spec['intervention_spec']
        print(f"\n[{i+1}/{len(pending)}] {trial_id}: {spec['description']}")

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

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            wm.restore()

    # === Analysis ===
    print("\n" + "=" * 60)
    print("ANALYSIS: Multi-Head Ablation Cascade")
    print("=" * 60)

    rows = db_conn.execute("""
        SELECT trial_id, intervention_spec, accuracy_delta, mcnemar_p_value,
               items_flipped_to_correct, items_flipped_to_incorrect
        FROM trials WHERE category='S4-MULTI' AND is_degenerate=0
        ORDER BY trial_id
    """).fetchall()

    print(f"\n  {'Trial':>25}  {'#Heads':>6}  {'Delta':>8}  {'p-val':>8}  {'+/-':>8}")
    print("  " + "-" * 65)
    for r in rows:
        spec = json.loads(r[1])
        n_heads = len(spec['heads'])
        print(f"  {r[0]:>25}  {n_heads:>6}  {r[2]:+6.1f}pp  {r[3]:>8.4f}  +{r[4]}/-{r[5]}")

    # Compare with full W3-0020 zero
    print(f"\n  Reference: W3-0020 (full L9 attn zero): +3.7pp")
    print(f"  Sum of 9 positive individual ablations: +1.6pp")

    if rows:
        # Check if multi-head ablation scales non-linearly
        deltas = [(len(json.loads(r[1])['heads']), r[2]) for r in rows]
        deltas.sort()
        print(f"\n  Scaling behavior:")
        for n, d in deltas:
            expected_linear = n * (1.6 / 9)  # linear extrapolation from sum
            ratio = d / expected_linear if expected_linear != 0 else float('inf')
            print(f"    {n} heads: {d:+.1f}pp (linear prediction: {expected_linear:+.1f}pp, ratio: {ratio:.1f}x)")

    db_conn.close()
    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()
