"""
Trial runner: applies an intervention, runs benchmark, computes stats, saves results.
Handles weight management, hook lifecycle, degenerate detection, and crash recovery.
"""
import json
import time
from datetime import datetime, timezone
from typing import Optional

import torch

import config
from benchmark import run_benchmark, evaluate_item_logprob
from weight_manager import WeightManager
from interventions import ActivationHook
from statistics import mcnemar_exact_test
from database import save_trial


def check_degeneracy(item_results: list[dict], threshold: float = 0.5) -> tuple[bool, str]:
    """Check for degenerate outputs per spec v0.4.

    Returns (is_degenerate, failure_mode).
    """
    if not item_results:
        return True, "no_results"

    # Log-prob degeneracy: identical logprobs for all options
    uniform_count = 0
    nan_count = 0
    for r in item_results:
        logprobs = r.get('logprobs', {})
        if not logprobs:
            continue
        values = list(logprobs.values())
        if any(v != v for v in values):  # NaN check
            nan_count += 1
        elif max(values) - min(values) < 1e-6:
            uniform_count += 1

    n = len(item_results)
    if nan_count > 0:
        return True, f"numerical_degeneracy: {nan_count} items with NaN/inf logprobs"
    if uniform_count >= n * threshold:
        return True, f"logprob_degeneracy: {uniform_count}/{n} items with uniform logprobs"

    return False, ""


def run_trial(
    model,
    tokenizer,
    items: list[dict],
    baseline_results: list[dict],
    wm: WeightManager,
    trial_id: str,
    category: str,
    intervention_spec: dict,
    weight_modifications: Optional[dict] = None,
    activation_hook: Optional[ActivationHook] = None,
    db_conn=None,
    device: str = "cuda",
) -> dict:
    """Run a single intervention trial.

    Args:
        model: the model (will be modified and restored)
        tokenizer: tokenizer
        items: benchmark items
        baseline_results: per-item baseline results for comparison
        wm: WeightManager for weight restore/verify
        trial_id: unique trial identifier
        category: intervention category (W1, W2, ..., A1, A2, A3)
        intervention_spec: full JSON-serializable intervention specification
        weight_modifications: dict of param_name -> tensor for Class W interventions
        activation_hook: ActivationHook instance for Class A interventions
        db_conn: SQLite connection for saving results
        device: torch device

    Returns:
        dict with trial results
    """
    start_time = time.time()
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        # Apply weight modifications if any
        if weight_modifications:
            wm.apply_weight_modification(weight_modifications)

        # Run early abort check (first 20% of items)
        n_early = max(1, len(items) // 5)
        early_results = []
        for item in items[:n_early]:
            r = evaluate_item_logprob(model, tokenizer, item, device)
            r['item_id'] = item['item_id']
            r['task'] = item['task']
            early_results.append(r)

        is_degenerate, failure_mode = check_degeneracy(early_results)
        if is_degenerate:
            wall_clock = time.time() - start_time
            trial_result = {
                'trial_id': trial_id,
                'category': category,
                'intervention_spec': intervention_spec,
                'is_degenerate': True,
                'accuracy': None,
                'accuracy_delta': None,
                'items_flipped_to_correct': None,
                'items_flipped_to_incorrect': None,
                'mcnemar_p_value': None,
                'bh_significant': None,
                'tier': None,
                'tier_justification': f"Degenerate: {failure_mode}",
                'wall_clock_seconds': wall_clock,
                'vram_peak_bytes': torch.cuda.max_memory_allocated() if device == "cuda" else None,
                'timestamp_utc': timestamp,
                'random_seed': config.RANDOM_SEED,
            }
            if db_conn:
                save_trial(db_conn, trial_result, [])
            return trial_result

        # Run remaining items
        remaining_results = []
        for item in items[n_early:]:
            r = evaluate_item_logprob(model, tokenizer, item, device)
            r['item_id'] = item['item_id']
            r['task'] = item['task']
            remaining_results.append(r)

        all_results = early_results + remaining_results

        # Compute statistics
        correct_count = sum(1 for r in all_results if r['correct'])
        n = len(all_results)
        accuracy = correct_count / n
        baseline_accuracy = sum(1 for r in baseline_results if r['correct']) / len(baseline_results)
        accuracy_delta_pp = (accuracy - baseline_accuracy) * 100

        # Count flips (McNemar table)
        items_flipped_to_correct = 0   # c: was wrong, now right
        items_flipped_to_incorrect = 0  # b: was right, now wrong

        baseline_map = {r['item_id']: r['correct'] for r in baseline_results}
        item_level_results = []

        for r in all_results:
            bl_correct = baseline_map.get(r['item_id'], False)
            int_correct = r['correct']

            if bl_correct and not int_correct:
                items_flipped_to_incorrect += 1
            elif not bl_correct and int_correct:
                items_flipped_to_correct += 1

            item_level_results.append({
                'trial_id': trial_id,
                'item_id': r['item_id'],
                'baseline_correct': bl_correct,
                'intervention_correct': int_correct,
                'baseline_logprobs': baseline_map.get(r['item_id'] + '_logprobs'),
                'intervention_logprobs': r.get('logprobs'),
            })

        # McNemar's test
        p_value = mcnemar_exact_test(
            b=items_flipped_to_incorrect,
            c=items_flipped_to_correct,
        )

        wall_clock = time.time() - start_time

        trial_result = {
            'trial_id': trial_id,
            'category': category,
            'intervention_spec': intervention_spec,
            'is_degenerate': False,
            'accuracy': accuracy,
            'accuracy_delta': accuracy_delta_pp,
            'items_flipped_to_correct': items_flipped_to_correct,
            'items_flipped_to_incorrect': items_flipped_to_incorrect,
            'mcnemar_p_value': p_value,
            'bh_significant': None,  # filled in after global BH correction
            'tier': None,  # filled in after BH
            'tier_justification': None,
            'wall_clock_seconds': wall_clock,
            'vram_peak_bytes': torch.cuda.max_memory_allocated() if device == "cuda" else None,
            'timestamp_utc': timestamp,
            'random_seed': config.RANDOM_SEED,
        }

        if db_conn:
            save_trial(db_conn, trial_result, item_level_results)

        return trial_result

    finally:
        # Always restore weights and remove hooks
        if weight_modifications:
            wm.restore()
            if not wm.verify():
                print(f"  CRITICAL: Weight verification failed after trial {trial_id}!")

        if activation_hook:
            activation_hook.remove_all()


def print_trial_summary(result: dict):
    """Print a one-line summary of a trial result."""
    if result['is_degenerate']:
        print(f"  {result['trial_id']}: DEGENERATE — {result['tier_justification']}")
    else:
        delta = result['accuracy_delta']
        flipped_c = result['items_flipped_to_correct']
        flipped_i = result['items_flipped_to_incorrect']
        p = result['mcnemar_p_value']
        print(f"  {result['trial_id']}: Δ={delta:+.1f}pp  +{flipped_c}/-{flipped_i}  p={p:.4f}  ({result['wall_clock_seconds']:.1f}s)")
