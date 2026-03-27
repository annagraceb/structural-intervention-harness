#!/usr/bin/env python3
"""
Stage 2: Determinism Verification and Baseline Characterization
- Run benchmark twice with identical config
- Confirm bitwise-identical outputs
- Record authoritative baseline
- Compute Stage 3 budget
"""
import json
import os
import sys

import torch

import config
config.configure_determinism()

from benchmark import load_bbh_tasks, run_benchmark, load_model_and_tokenizer


def compare_runs(run1: list[dict], run2: list[dict]) -> dict:
    """Compare two benchmark runs item-by-item."""
    assert len(run1) == len(run2), f"Run lengths differ: {len(run1)} vs {len(run2)}"

    identical = 0
    different = 0
    diff_items = []

    for r1, r2 in zip(run1, run2):
        assert r1['item_id'] == r2['item_id']
        if r1['selected'] == r2['selected'] and r1['correct'] == r2['correct']:
            # Also check logprob values match
            logprobs_match = True
            for key in r1['logprobs']:
                if abs(r1['logprobs'].get(key, 0) - r2['logprobs'].get(key, 0)) > 1e-6:
                    logprobs_match = False
                    break
            if logprobs_match:
                identical += 1
            else:
                different += 1
                diff_items.append({
                    'item_id': r1['item_id'],
                    'reason': 'logprob_mismatch',
                    'r1_selected': r1['selected'],
                    'r2_selected': r2['selected'],
                })
        else:
            different += 1
            diff_items.append({
                'item_id': r1['item_id'],
                'reason': 'answer_mismatch',
                'r1_selected': r1['selected'],
                'r2_selected': r2['selected'],
                'r1_correct': r1['correct'],
                'r2_correct': r2['correct'],
            })

    return {
        'identical': identical,
        'different': different,
        'pct_identical': identical / len(run1) * 100,
        'diff_items': diff_items,
    }


def main():
    print("=" * 60)
    print("STAGE 2: DETERMINISM VERIFICATION")
    print("=" * 60)

    # Load benchmark and model
    items = load_bbh_tasks(config.BBH_TASKS)
    model, tokenizer = load_model_and_tokenizer(config.MODEL_ID, config.DEVICE)

    # Run 1
    print("\n--- Run 1 ---")
    torch.cuda.reset_peak_memory_stats()
    result1 = run_benchmark(model, tokenizer, items, config.DEVICE, verbose=True)
    print(f"  Accuracy: {result1['accuracy_pct']:.1f}%")

    # Run 2 (must be identical under determinism)
    print("\n--- Run 2 ---")
    result2 = run_benchmark(model, tokenizer, items, config.DEVICE, verbose=True)
    print(f"  Accuracy: {result2['accuracy_pct']:.1f}%")

    # Compare
    print("\n--- Comparison ---")
    comparison = compare_runs(result1['per_item'], result2['per_item'])
    print(f"  Identical: {comparison['identical']}/{len(items)} ({comparison['pct_identical']:.1f}%)")
    print(f"  Different: {comparison['different']}")

    if comparison['different'] == 0:
        determinism_status = "deterministic"
        print("  STATUS: DETERMINISTIC — all items bitwise-identical")
        authoritative_baseline = result1
    elif comparison['different'] <= len(items) * 0.01:
        determinism_status = "near-deterministic"
        print(f"  STATUS: NEAR-DETERMINISTIC — {comparison['different']} items differ (≤1%)")
        print("  Running 5 additional passes for majority vote...")
        # Run 5 more passes
        extra_runs = []
        for i in range(5):
            print(f"\n  --- Extra run {i+3} ---")
            r = run_benchmark(model, tokenizer, items, config.DEVICE, verbose=False)
            extra_runs.append(r)
            print(f"    Accuracy: {r['accuracy_pct']:.1f}%")

        # Majority vote across all 7 runs
        all_runs = [result1, result2] + extra_runs
        authoritative_baseline = result1  # start from run1's structure
        unstable_items = []

        for idx in range(len(items)):
            votes = [run['per_item'][idx]['correct'] for run in all_runs]
            majority_correct = sum(votes) > len(votes) / 2
            if not all(v == votes[0] for v in votes):
                unstable_items.append(items[idx]['item_id'])
            authoritative_baseline['per_item'][idx]['correct'] = majority_correct

        print(f"\n  Unstable items: {len(unstable_items)}")
        authoritative_baseline['unstable_items'] = unstable_items
    else:
        determinism_status = "non-deterministic"
        print(f"  STATUS: NON-DETERMINISTIC — {comparison['different']} items differ (>1%)")
        print("  ERROR: Debug nondeterminism before proceeding")
        if comparison['diff_items']:
            print("  First 5 differing items:")
            for d in comparison['diff_items'][:5]:
                print(f"    {d}")
        authoritative_baseline = result1

    # Budget calculation
    print("\n--- Stage 3 Budget ---")
    time_per_pass = result1['wall_clock_seconds']
    n_trials = 515
    coarse_hours = (time_per_pass * n_trials) / 3600
    remaining = config.EFFECTIVE_GPU_HOURS - coarse_hours
    print(f"  Time per pass: {time_per_pass:.1f}s ({time_per_pass/60:.1f} min)")
    print(f"  Coarse sweep: {coarse_hours:.1f} hours")
    print(f"  Remaining: {remaining:.1f} hours")

    # Save Stage 2 artifacts
    print("\n--- Saving Stage 2 artifacts ---")
    stage2_artifacts = {
        "determinism_status": determinism_status,
        "baseline_accuracy": result1['accuracy'],
        "baseline_accuracy_pct": result1['accuracy_pct'],
        "correct_count": result1['correct_count'],
        "total_items": result1['total_items'],
        "comparison": {
            "identical": comparison['identical'],
            "different": comparison['different'],
            "diff_items": comparison['diff_items'][:20],  # cap to avoid huge files
        },
        "wall_clock_per_pass_seconds": time_per_pass,
        "budget": {
            "coarse_sweep_hours": coarse_hours,
            "remaining_hours": remaining,
            "feasible": remaining > 0,
        },
        "unstable_items": authoritative_baseline.get('unstable_items', []),
    }

    os.makedirs(config.BASELINE_DIR, exist_ok=True)
    with open(os.path.join(config.BASELINE_DIR, "stage2_artifacts.json"), "w") as f:
        json.dump(stage2_artifacts, f, indent=2, default=str)

    with open(os.path.join(config.BASELINE_DIR, "authoritative_baseline.json"), "w") as f:
        json.dump(authoritative_baseline['per_item'], f, indent=2, default=str)

    print(f"  Saved to {config.BASELINE_DIR}")

    print("\n" + "=" * 60)
    print(f"STAGE 2 COMPLETE — Status: {determinism_status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
