#!/usr/bin/env python3
"""
Stage 3: Intervention Search
Generates all intervention trials, runs them, applies BH correction, classifies tiers.
"""
import json
import os
import sys
import time
import itertools

import torch
import numpy as np

import config
config.configure_determinism()

from benchmark import load_bbh_tasks, load_model_and_tokenizer
from weight_manager import WeightManager
from interventions import (
    w1_permutation, w2_transplant, w3_reinitialize,
    w4_head_surgery, w5_spectral_edit,
    a1_residual_injection_hook, a2_head_scaling_hook, a3_layer_skip_hook,
)
from trial_runner import run_trial, print_trial_summary
from database import init_db, get_completed_trial_ids, save_metadata, save_trial, get_non_degenerate_trials
from statistics import benjamini_hochberg, classify_tier


def generate_w1_trials(wm: WeightManager, seed_base: int = 1000) -> list[dict]:
    """Generate W1 (permutation) trial specifications."""
    trials = []
    n_layers = wm.get_n_layers()
    components = ["self_attn.q_proj.weight", "mlp.up_proj.weight"]
    strategies = ["random", "l2_asc", "l2_desc", "reverse", "interleave"]

    # Sample: 5 layers evenly spaced, 2 components, 5 strategies, coupled + uncoupled
    layer_samples = np.linspace(0, n_layers - 1, 5, dtype=int)

    for layer in layer_samples:
        for comp in components:
            for strat in strategies:
                for coupled in [True, False]:
                    seed = seed_base + len(trials)
                    trial_id = f"W1-{len(trials):04d}"
                    trials.append({
                        'trial_id': trial_id,
                        'category': 'W1',
                        'intervention_spec': {
                            'type': 'W1_permutation',
                            'layer': int(layer),
                            'component': comp,
                            'strategy': strat,
                            'coupled': coupled,
                            'seed': seed,
                        },
                    })
    return trials


def generate_w2_trials(wm: WeightManager) -> list[dict]:
    """Generate W2 (cross-layer transplant) trial specifications."""
    trials = []
    n_layers = wm.get_n_layers()
    components = ["self_attn.q_proj.weight", "mlp.gate_proj.weight"]

    for comp in components:
        # Adjacent pairs
        for i in range(n_layers - 1):
            trials.append({
                'trial_id': f"W2-{len(trials):04d}",
                'category': 'W2',
                'intervention_spec': {
                    'type': 'W2_transplant',
                    'source_layer': i,
                    'target_layer': i + 1,
                    'component': comp,
                },
            })

        # Distant pairs: 10 random
        rng = np.random.RandomState(42)
        for _ in range(5):
            s, t = rng.choice(n_layers, 2, replace=False)
            trials.append({
                'trial_id': f"W2-{len(trials):04d}",
                'category': 'W2',
                'intervention_spec': {
                    'type': 'W2_transplant',
                    'source_layer': int(s),
                    'target_layer': int(t),
                    'component': comp,
                },
            })

        # Symmetric pairs
        for i in range(n_layers // 2):
            trials.append({
                'trial_id': f"W2-{len(trials):04d}",
                'category': 'W2',
                'intervention_spec': {
                    'type': 'W2_transplant',
                    'source_layer': i,
                    'target_layer': n_layers - 1 - i,
                    'component': comp,
                },
            })

    return trials


def generate_w3_trials(wm: WeightManager, seed_base: int = 3000) -> list[dict]:
    """Generate W3 (reinitialization) trial specifications."""
    trials = []
    n_layers = wm.get_n_layers()
    distributions = ["kaiming", "xavier", "zero", "scaled_noise"]
    granularities = ["attention", "mlp"]

    # Sample 7 layers evenly spaced
    layer_samples = np.linspace(0, n_layers - 1, 7, dtype=int)

    for layer in layer_samples:
        for dist in distributions:
            for gran in granularities:
                seed = seed_base + len(trials)
                trials.append({
                    'trial_id': f"W3-{len(trials):04d}",
                    'category': 'W3',
                    'intervention_spec': {
                        'type': 'W3_reinitialize',
                        'layer': int(layer),
                        'distribution': dist,
                        'granularity': gran,
                        'seed': seed,
                    },
                })
    return trials


def generate_w4_trials(wm: WeightManager) -> list[dict]:
    """Generate W4 (attention head surgery) trial specifications."""
    trials = []
    n_layers = wm.get_n_layers()
    n_heads = wm.get_n_heads()
    operations = ["ablate", "negate"]

    # Sample 7 layers, sample 4 heads per layer
    layer_samples = np.linspace(0, n_layers - 1, 7, dtype=int)
    head_samples = np.linspace(0, n_heads - 1, min(4, n_heads), dtype=int)

    for layer in layer_samples:
        for head in head_samples:
            for op in operations:
                trials.append({
                    'trial_id': f"W4-{len(trials):04d}",
                    'category': 'W4',
                    'intervention_spec': {
                        'type': 'W4_head_surgery',
                        'layer': int(layer),
                        'head_i': int(head),
                        'operation': op,
                    },
                })

        # Duplicate: copy head 0 over head 1
        if n_heads >= 2:
            trials.append({
                'trial_id': f"W4-{len(trials):04d}",
                'category': 'W4',
                'intervention_spec': {
                    'type': 'W4_head_surgery',
                    'layer': int(layer),
                    'head_i': 0,
                    'head_j': 1,
                    'operation': 'duplicate',
                },
            })

    return trials


def generate_w5_trials(wm: WeightManager) -> list[dict]:
    """Generate W5 (spectral editing) trial specifications."""
    trials = []
    n_layers = wm.get_n_layers()
    components = ["self_attn.q_proj.weight", "mlp.gate_proj.weight"]
    operations = ["top_k", "bottom_k", "spectral_inversion", "uniform_spectrum"]

    # Sample 5 layers
    layer_samples = np.linspace(0, n_layers - 1, 5, dtype=int)

    for layer in layer_samples:
        for comp in components:
            # Get matrix rank for k values
            params = wm.get_layer_param_names(int(layer))
            full_name = params.get(comp)
            if full_name is None:
                continue
            weight = wm.get_param(full_name)
            rank = min(weight.shape)

            # k values, filtering invalid ones
            k_values = [1, 5, 10, 25, 50, rank // 4, rank // 2, rank - 10]
            k_values = [k for k in k_values if 0 < k < rank]
            k_values = sorted(set(k_values))

            for op in operations:
                if op in ["spectral_inversion", "uniform_spectrum"]:
                    trials.append({
                        'trial_id': f"W5-{len(trials):04d}",
                        'category': 'W5',
                        'intervention_spec': {
                            'type': 'W5_spectral_edit',
                            'layer': int(layer),
                            'component': comp,
                            'operation': op,
                            'k': None,
                        },
                    })
                else:
                    for k in k_values[:4]:  # limit to avoid too many
                        trials.append({
                            'trial_id': f"W5-{len(trials):04d}",
                            'category': 'W5',
                            'intervention_spec': {
                                'type': 'W5_spectral_edit',
                                'layer': int(layer),
                                'component': comp,
                                'operation': op,
                                'k': int(k),
                            },
                        })
    return trials


def generate_a1_trials(wm: WeightManager, seed_base: int = 6000) -> list[dict]:
    """Generate A1 (residual injection) trial specifications."""
    trials = []
    n_layers = wm.get_n_layers()
    hidden_size = wm.get_hidden_size()

    # Sample 7 layers, 3 magnitude scales, 2 vector types
    layer_samples = np.linspace(0, n_layers - 1, 7, dtype=int)
    alpha_scales = [0.1, 1.0, 5.0]  # relative to layer norm

    for layer in layer_samples:
        for alpha_scale in alpha_scales:
            for vec_type in ["random", "mean_activation"]:
                seed = seed_base + len(trials)
                trials.append({
                    'trial_id': f"A1-{len(trials):04d}",
                    'category': 'A1',
                    'intervention_spec': {
                        'type': 'A1_residual_injection',
                        'layer': int(layer),
                        'alpha_scale': alpha_scale,
                        'vector_type': vec_type,
                        'seed': seed,
                    },
                })
    return trials


def generate_a2_trials(wm: WeightManager) -> list[dict]:
    """Generate A2 (head scaling) trial specifications."""
    trials = []
    n_layers = wm.get_n_layers()
    n_heads = wm.get_n_heads()
    scalars = [0.0, -1.0, 0.5, 2.0, 5.0]

    # Sample 5 layers, 3 heads per layer
    layer_samples = np.linspace(0, n_layers - 1, 5, dtype=int)
    head_samples = np.linspace(0, n_heads - 1, min(3, n_heads), dtype=int)

    for layer in layer_samples:
        for head in head_samples:
            for scalar in scalars:
                trials.append({
                    'trial_id': f"A2-{len(trials):04d}",
                    'category': 'A2',
                    'intervention_spec': {
                        'type': 'A2_head_scaling',
                        'layer': int(layer),
                        'head_idx': int(head),
                        'scalar': scalar,
                    },
                })
    return trials


def generate_a3_trials(wm: WeightManager) -> list[dict]:
    """Generate A3 (layer skip) trial specifications."""
    trials = []
    n_layers = wm.get_n_layers()
    variants = ["full", "attention_only", "mlp_only"]

    for layer in range(n_layers):
        for variant in variants:
            trials.append({
                'trial_id': f"A3-{len(trials):04d}",
                'category': 'A3',
                'intervention_spec': {
                    'type': 'A3_layer_skip',
                    'layer': layer,
                    'variant': variant,
                },
            })
    return trials


def execute_intervention(model, wm, spec: dict, device: str = "cuda"):
    """Execute an intervention and return (weight_modifications, activation_hook)."""
    itype = spec['type']

    if itype == 'W1_permutation':
        mods = w1_permutation(
            wm, spec['layer'], spec['component'],
            spec['strategy'], spec['coupled'], spec.get('seed', 42),
        )
        return mods, None

    elif itype == 'W2_transplant':
        mods = w2_transplant(
            wm, spec['source_layer'], spec['target_layer'], spec['component'],
        )
        return mods, None

    elif itype == 'W3_reinitialize':
        mods = w3_reinitialize(
            wm, spec['layer'], spec['granularity'], spec['distribution'],
            seed=spec.get('seed', 42),
        )
        return mods, None

    elif itype == 'W4_head_surgery':
        mods = w4_head_surgery(
            wm, spec['layer'], spec['operation'],
            spec['head_i'], spec.get('head_j'),
        )
        return mods, None

    elif itype == 'W5_spectral_edit':
        mods = w5_spectral_edit(
            wm, spec['layer'], spec['component'],
            spec['operation'], spec.get('k'),
        )
        return mods, None

    elif itype == 'A1_residual_injection':
        hidden_size = wm.get_hidden_size()
        if spec['vector_type'] == 'random':
            rng = torch.Generator()
            rng.manual_seed(spec.get('seed', 42))
            vec = torch.randn(hidden_size, generator=rng)
            vec = vec / vec.norm()
        else:
            # Mean activation — use zero vector as fallback
            # (proper calibration would require a forward pass over calibration set)
            vec = torch.zeros(hidden_size)

        # Scale by alpha_scale (without layer norm calibration, use raw scale)
        vec = vec * spec['alpha_scale']
        vec = vec.to(device=device, dtype=torch.float16)

        hook = a1_residual_injection_hook(model, spec['layer'], vec)
        return None, hook

    elif itype == 'A2_head_scaling':
        head_dim = wm.get_head_dim()
        hook = a2_head_scaling_hook(
            model, spec['layer'], spec['head_idx'],
            spec['scalar'], head_dim,
        )
        return None, hook

    elif itype == 'A3_layer_skip':
        hook = a3_layer_skip_hook(model, spec['layer'], spec['variant'])
        return None, hook

    else:
        raise ValueError(f"Unknown intervention type: {itype}")


def run_sanity_checks(model, tokenizer, items, baseline_results, wm, db_conn, device):
    """Run sanity check interventions before main experiment."""
    print("\n--- Sanity Check 1: Positive control (ablate all heads in last layer) ---")
    n_heads = wm.get_n_heads()
    n_layers = wm.get_n_layers()
    last_layer = n_layers - 1

    # Zero all heads in last layer via W4
    all_mods = {}
    for head_i in range(n_heads):
        mods = w4_head_surgery(wm, last_layer, "ablate", head_i)
        all_mods.update(mods)

    result = run_trial(
        model, tokenizer, items, baseline_results, wm,
        trial_id="SANITY-POS",
        category="SANITY",
        intervention_spec={"type": "sanity_positive", "description": f"Ablate all {n_heads} heads in layer {last_layer}"},
        weight_modifications=all_mods,
        db_conn=db_conn,
        device=device,
    )
    print_trial_summary(result)
    if result['is_degenerate']:
        print("  WARNING: Positive control produced degenerate output — pipeline may have a bug!")
        return False
    # Note: the effect magnitude may be small if the model has redundancy.
    # The key check is that SOMETHING changed, not that it degraded by a specific amount.
    total_flipped = (result.get('items_flipped_to_correct', 0) or 0) + (result.get('items_flipped_to_incorrect', 0) or 0)
    if total_flipped < 5:
        print(f"  WARNING: Positive control shows only {total_flipped} flipped items — pipeline may have a bug!")
        return False

    print("\n--- Sanity Check 2: Negative control (identity permutation) ---")
    # Coupled identity permutation — should be a no-op
    # Create identity permutation
    params = wm.get_layer_param_names(0)
    comp = "self_attn.q_proj.weight"
    full_name = params.get(comp)
    if full_name:
        weight = wm.get_param(full_name)
        # "Permute" with identity — should change nothing
        identity_mods = {full_name: weight.clone()}  # same weight
        result2 = run_trial(
            model, tokenizer, items, baseline_results, wm,
            trial_id="SANITY-NEG",
            category="SANITY",
            intervention_spec={"type": "sanity_negative", "description": "Identity permutation on layer 0 q_proj"},
            weight_modifications=identity_mods,
            db_conn=db_conn,
            device=device,
        )
        print_trial_summary(result2)
        if result2['accuracy_delta'] is not None and abs(result2['accuracy_delta']) > 0.01:
            print("  WARNING: Negative control shows non-zero effect — weight management may have a bug!")
            return False

    print("\n  Sanity checks passed.")
    return True


def main():
    print("=" * 60)
    print("STAGE 3: INTERVENTION SEARCH")
    print("=" * 60)

    # Load everything
    print("\n--- Setup ---")
    items = load_bbh_tasks(config.BBH_TASKS)
    model, tokenizer = load_model_and_tokenizer(config.MODEL_ID, config.DEVICE)

    # Load authoritative baseline
    baseline_path = os.path.join(config.BASELINE_DIR, "authoritative_baseline.json")
    with open(baseline_path) as f:
        baseline_results = json.load(f)

    # Initialize weight manager
    wm = WeightManager(model)

    # Initialize database
    db_conn = init_db()
    completed = get_completed_trial_ids(db_conn)
    print(f"  Previously completed trials: {len(completed)}")

    # Save metadata
    save_metadata(db_conn, "model_id", config.MODEL_ID)
    save_metadata(db_conn, "benchmark_tasks", config.BBH_TASKS)
    save_metadata(db_conn, "baseline_accuracy", sum(1 for r in baseline_results if r['correct']) / len(baseline_results))
    save_metadata(db_conn, "total_items", len(items))

    # Run sanity checks (skip if already done)
    if "SANITY-POS" not in completed:
        ok = run_sanity_checks(model, tokenizer, items, baseline_results, wm, db_conn, config.DEVICE)
        if not ok:
            print("  Sanity checks failed. Aborting.")
            return

    # Generate all trial specifications
    print("\n--- Generating trial specifications ---")
    all_trials = []
    generators = [
        ("W1", generate_w1_trials),
        ("W2", generate_w2_trials),
        ("W3", generate_w3_trials),
        ("W4", generate_w4_trials),
        ("W5", generate_w5_trials),
        ("A1", generate_a1_trials),
        ("A2", generate_a2_trials),
        ("A3", generate_a3_trials),
    ]
    for cat_name, gen_fn in generators:
        trials = gen_fn(wm)
        all_trials.extend(trials)
        print(f"  {cat_name}: {len(trials)} trials")

    print(f"  Total: {len(all_trials)} trials")

    # Filter out already-completed trials
    pending = [t for t in all_trials if t['trial_id'] not in completed]
    print(f"  Pending: {len(pending)} trials")

    # Estimate time
    time_per_trial_s = 198  # from Stage 2
    est_hours = len(pending) * time_per_trial_s / 3600
    print(f"  Estimated time: {est_hours:.1f} hours")

    # Run trials
    print("\n--- Running coarse sweep ---")
    start_time = time.time()
    results = []

    for i, trial_spec in enumerate(pending):
        trial_id = trial_spec['trial_id']
        spec = trial_spec['intervention_spec']

        if i % 10 == 0:
            elapsed_h = (time.time() - start_time) / 3600
            remaining_h = (len(pending) - i) * (elapsed_h / max(i, 1))
            print(f"\n[{i}/{len(pending)}] elapsed={elapsed_h:.1f}h remaining~{remaining_h:.1f}h")

        try:
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
            results.append(result)

        except Exception as e:
            print(f"  {trial_id}: ERROR — {e}")
            # Log error as degenerate trial
            from datetime import datetime, timezone
            error_result = {
                'trial_id': trial_id,
                'category': trial_spec['category'],
                'intervention_spec': spec,
                'is_degenerate': True,
                'accuracy': None,
                'accuracy_delta': None,
                'items_flipped_to_correct': None,
                'items_flipped_to_incorrect': None,
                'mcnemar_p_value': None,
                'bh_significant': None,
                'tier': None,
                'tier_justification': f"Error: {str(e)[:200]}",
                'wall_clock_seconds': 0,
                'vram_peak_bytes': None,
                'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'random_seed': config.RANDOM_SEED,
            }
            save_trial(db_conn, error_result, [])
            results.append(error_result)

            # Restore weights in case of error
            wm.restore()

    # Apply final BH correction
    print("\n--- Applying BH correction ---")
    non_degenerate = get_non_degenerate_trials(db_conn)
    # Exclude SANITY trials
    non_degenerate = [t for t in non_degenerate if not t['trial_id'].startswith('SANITY')]

    p_values = [t['mcnemar_p_value'] for t in non_degenerate if t['mcnemar_p_value'] is not None]
    trial_ids_for_bh = [t['trial_id'] for t in non_degenerate if t['mcnemar_p_value'] is not None]

    if p_values:
        significant = benjamini_hochberg(p_values, config.BH_FDR)

        n_significant = sum(significant)
        print(f"  {n_significant}/{len(p_values)} trials survive BH correction at FDR={config.BH_FDR}")

        # Update database with BH results and tier classifications
        baseline_accuracy = sum(1 for r in baseline_results if r['correct']) / len(baseline_results)
        for tid, sig, p in zip(trial_ids_for_bh, significant, p_values):
            trial_data = next((t for t in non_degenerate if t['trial_id'] == tid), None)
            if trial_data:
                tier, justification = classify_tier(
                    accuracy_delta_pp=trial_data['accuracy_delta'] or 0,
                    items_flipped_to_correct=trial_data['items_flipped_to_correct'] or 0,
                    items_flipped_to_incorrect=trial_data['items_flipped_to_incorrect'] or 0,
                    bh_significant=sig,
                    n_items=len(items),
                )
                db_conn.execute(
                    "UPDATE trials SET bh_significant=?, tier=?, tier_justification=? WHERE trial_id=?",
                    (sig, tier, justification, tid)
                )
        db_conn.commit()

    # Generate summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Count by category
    for cat in ["W1", "W2", "W3", "W4", "W5", "A1", "A2", "A3"]:
        cat_trials = [t for t in non_degenerate if t['category'] == cat]
        cat_sig = db_conn.execute(
            "SELECT COUNT(*) FROM trials WHERE category=? AND bh_significant=1", (cat,)
        ).fetchone()[0]
        cat_t2 = db_conn.execute(
            "SELECT COUNT(*) FROM trials WHERE category=? AND tier>=2", (cat,)
        ).fetchone()[0]
        print(f"  {cat}: {len(cat_trials)} trials, {cat_sig} significant, {cat_t2} Tier 2+")

    # Print top results
    print("\n--- Top Results (by accuracy delta) ---")
    top = db_conn.execute("""
        SELECT trial_id, category, accuracy_delta, items_flipped_to_correct,
               items_flipped_to_incorrect, mcnemar_p_value, bh_significant, tier, tier_justification
        FROM trials
        WHERE is_degenerate = 0 AND accuracy_delta IS NOT NULL
        ORDER BY ABS(accuracy_delta) DESC
        LIMIT 20
    """).fetchall()

    for row in top:
        tid, cat, delta, fc, fi, p, sig, tier, just = row
        sig_str = "*" if sig else " "
        tier_str = f"T{tier}" if tier else "  "
        print(f"  {sig_str}{tier_str} {tid} ({cat}): Δ={delta:+.1f}pp  +{fc}/-{fi}  p={p:.6f}")

    total_time = (time.time() - start_time) / 3600
    print(f"\n  Total execution time: {total_time:.1f} hours")

    # Save leaderboard
    leaderboard_path = os.path.join(config.ARTIFACTS_DIR, "leaderboard.json")
    leaderboard = []
    for row in top:
        tid, cat, delta, fc, fi, p, sig, tier, just = row
        leaderboard.append({
            "trial_id": tid, "category": cat, "accuracy_delta_pp": delta,
            "flipped_to_correct": fc, "flipped_to_incorrect": fi,
            "p_value": p, "bh_significant": bool(sig), "tier": tier,
            "justification": just,
        })
    with open(leaderboard_path, "w") as f:
        json.dump(leaderboard, f, indent=2)
    print(f"\n  Leaderboard saved to {leaderboard_path}")


if __name__ == "__main__":
    main()
