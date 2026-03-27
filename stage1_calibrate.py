#!/usr/bin/env python3
"""
Stage 1: Calibration
- Download BBH tasks
- Load candidate model
- Run baseline accuracy
- Profile VRAM and timing
- Run power analysis
- Freeze Stage 1 artifacts
"""
import json
import os
import sys
import time
import math

import torch
from scipy import stats

import config
from benchmark import load_bbh_tasks, run_benchmark, load_model_and_tokenizer

config.configure_determinism()


def power_analysis(n_items: int, baseline_acc: float, n_interventions: int, fdr: float = 0.05):
    """Compute minimum detectable effect size for McNemar's test with BH correction.

    For the top-ranked intervention, BH requires p < fdr / n_interventions.
    McNemar's exact test: under H0, discordant pairs are Binomial(n_discordant, 0.5).
    We need P(X <= min(b,c)) * 2 < threshold.
    """
    threshold_top = fdr / n_interventions  # hardest to pass
    threshold_mid = fdr * (n_interventions // 2) / n_interventions  # median rank

    results = {}
    for label, threshold in [("top_ranked", threshold_top), ("median_ranked", threshold_mid)]:
        # For a purely constructive intervention (b=0), p = 2 * 0.5^n
        # Solve: 2 * 0.5^n < threshold => n > log2(2/threshold)
        min_flips_b0 = math.ceil(math.log2(2.0 / threshold))

        # For b=1: p = 2 * (1+n) * 0.5^n < threshold
        min_flips_b1 = None
        for n in range(1, 200):
            p = 2 * (1 + n) * (0.5 ** n)
            if p < threshold:
                min_flips_b1 = n
                break

        # For b=2: p = 2 * [1 + n + n*(n-1)/2] * 0.5^n < threshold
        min_flips_b2 = None
        for n in range(1, 200):
            p = 2 * (1 + n + n * (n - 1) / 2) * (0.5 ** n)
            if p < threshold:
                min_flips_b2 = n
                break

        results[label] = {
            "bh_threshold": threshold,
            "min_discordant_b0": min_flips_b0,
            "min_discordant_b1": min_flips_b1,
            "min_discordant_b2": min_flips_b2,
            "min_accuracy_delta_b0_pp": (min_flips_b0 / n_items) * 100,
        }

    # Check if 3pp is achievable
    items_for_3pp = math.ceil(0.03 * n_items)
    results["tier2_feasibility"] = {
        "items_for_3pp": items_for_3pp,
        "min_flips_needed_top": results["top_ranked"]["min_discordant_b0"],
        "feasible": items_for_3pp >= results["top_ranked"]["min_discordant_b0"],
    }

    return results


def inspect_model_architecture(model):
    """Verify model architecture meets spec requirements."""
    info = {}

    # Get layer structure
    named_modules = dict(model.named_modules())

    # Find the transformer layers
    layers = None
    for name, module in named_modules.items():
        if hasattr(module, '__len__') and len(list(module.children())) > 5:
            if 'layer' in name.lower() or 'block' in name.lower():
                layers = module
                info['layers_path'] = name
                break

    # Check for separate Q, K, V projections
    sample_layer = None
    for name, module in named_modules.items():
        if 'q_proj' in name:
            info['has_separate_qkv'] = True
            # Get the layer that contains this
            parts = name.split('.')
            for i, p in enumerate(parts):
                if p.isdigit():
                    info['layer_index_position'] = i
                    break
            sample_layer = name.rsplit('.q_proj', 1)[0]
            break
        elif 'qkv_proj' in name or 'qkv' in name.split('.')[-1]:
            info['has_separate_qkv'] = False
            break

    if sample_layer:
        # Check all projection matrices in the sample layer
        projections = {}
        for name, module in named_modules.items():
            if name.startswith(sample_layer) and hasattr(module, 'weight'):
                short_name = name[len(sample_layer) + 1:]
                projections[short_name] = {
                    'shape': list(module.weight.shape),
                    'dtype': str(module.weight.dtype),
                }
        info['projections'] = projections

    # Count layers
    n_layers = 0
    for name, _ in named_modules.items():
        if '.0.self_attn' in name or '.0.attention' in name:
            # Found layer 0, now count
            prefix = name.split('.0.')[0]
            for name2, _ in named_modules.items():
                if name2.startswith(prefix + '.') and '.self_attn' in name2:
                    n_layers += 1
            # Deduplicate — each layer has multiple submodules with self_attn
            break

    # Better counting: look for sequential layers
    layer_indices = set()
    for name, _ in named_modules.items():
        parts = name.split('.')
        for i, p in enumerate(parts):
            if p.isdigit() and i > 0 and ('layer' in parts[i-1].lower() or parts[i-1] == 'layers' or parts[i-1] == 'h'):
                layer_indices.add(int(p))

    if layer_indices:
        info['n_layers'] = max(layer_indices) + 1
    else:
        info['n_layers'] = n_layers

    # Get model config
    model_config = model.config
    info['hidden_size'] = getattr(model_config, 'hidden_size', None)
    info['num_attention_heads'] = getattr(model_config, 'num_attention_heads', None)
    info['num_kv_heads'] = getattr(model_config, 'num_key_value_heads', None)
    info['intermediate_size'] = getattr(model_config, 'intermediate_size', None)
    info['vocab_size'] = getattr(model_config, 'vocab_size', None)
    info['head_dim'] = info['hidden_size'] // info['num_attention_heads'] if info['hidden_size'] and info['num_attention_heads'] else None
    info['is_gqa'] = info['num_kv_heads'] is not None and info['num_kv_heads'] != info['num_attention_heads']

    return info


def main():
    print("=" * 60)
    print("STAGE 1: CALIBRATION")
    print("=" * 60)

    # Step 1: Load BBH tasks
    print("\n--- Loading BBH tasks ---")
    items = load_bbh_tasks(config.BBH_TASKS)
    print(f"Total items: {len(items)}")

    task_counts = {}
    for item in items:
        task_counts[item["task"]] = task_counts.get(item["task"], 0) + 1
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count} items")

    if len(items) < config.MIN_BENCHMARK_ITEMS:
        print(f"WARNING: {len(items)} items < {config.MIN_BENCHMARK_ITEMS} minimum. Need more tasks.")
        # Try adding more tasks
        extra_tasks = [
            "logical_deduction_three_objects",
            "tracking_shuffled_objects_three_objects",
            "date_understanding",
            "web_of_lies",
            "navigate",
        ]
        for task in extra_tasks:
            if task not in config.BBH_TASKS:
                print(f"  Adding {task}...")
                try:
                    extra = load_bbh_tasks([task])
                    items.extend(extra)
                    task_counts[task] = len(extra)
                    print(f"    Added {len(extra)} items (total: {len(items)})")
                    if len(items) >= config.MIN_BENCHMARK_ITEMS:
                        break
                except Exception as e:
                    print(f"    Failed: {e}")

    print(f"\nFinal benchmark: {len(items)} items across {len(task_counts)} tasks")

    # Step 2: Load model
    print("\n--- Loading model ---")
    model, tokenizer = load_model_and_tokenizer(config.MODEL_ID, config.DEVICE)

    # Step 3: Inspect architecture
    print("\n--- Architecture verification ---")
    arch_info = inspect_model_architecture(model)
    for k, v in arch_info.items():
        if k != 'projections':
            print(f"  {k}: {v}")
    if 'projections' in arch_info:
        print("  Projections in sample layer:")
        for name, info in arch_info['projections'].items():
            print(f"    {name}: {info['shape']} ({info['dtype']})")

    # Step 4: Run quick calibration (first 50 items for speed estimate)
    print("\n--- Quick calibration (50 items) ---")
    quick_results = run_benchmark(model, tokenizer, items, config.DEVICE, max_items=50, verbose=True)
    print(f"  Quick accuracy: {quick_results['accuracy_pct']:.1f}%")
    print(f"  Time per item: {quick_results['seconds_per_item']:.3f}s")
    print(f"  Estimated full pass: {quick_results['seconds_per_item'] * len(items) / 60:.1f} min")

    # Step 5: Run full baseline
    print("\n--- Full baseline evaluation ---")
    full_results = run_benchmark(model, tokenizer, items, config.DEVICE, verbose=True)
    print(f"\n  Baseline accuracy: {full_results['accuracy_pct']:.1f}%")
    print(f"  Correct: {full_results['correct_count']}/{full_results['total_items']}")
    print(f"  Degenerate: {full_results['degenerate_count']}")
    print(f"  Wall clock: {full_results['wall_clock_seconds']:.1f}s ({full_results['wall_clock_seconds']/60:.1f} min)")
    print(f"  Per item: {full_results['seconds_per_item']:.3f}s")

    # Step 6: VRAM profile
    print("\n--- VRAM profile ---")
    if config.DEVICE == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Peak: {peak:.2f} GB")
        print(f"  Total: {total:.2f} GB")
        print(f"  Headroom: {total - peak:.2f} GB")

    # Step 7: Power analysis
    print("\n--- Power analysis ---")
    n_planned_interventions = 515
    power = power_analysis(
        n_items=full_results['total_items'],
        baseline_acc=full_results['accuracy'],
        n_interventions=n_planned_interventions,
        fdr=config.BH_FDR,
    )
    for label, results in power.items():
        print(f"  {label}:")
        for k, v in results.items():
            print(f"    {k}: {v}")

    # Step 8: Budget estimate
    print("\n--- Budget estimate ---")
    time_per_pass_hours = full_results['wall_clock_seconds'] / 3600
    total_coarse_hours = time_per_pass_hours * n_planned_interventions
    print(f"  Time per pass: {time_per_pass_hours*60:.1f} min")
    print(f"  Coarse sweep ({n_planned_interventions} trials): {total_coarse_hours:.1f} hours")
    print(f"  Effective budget: {config.EFFECTIVE_GPU_HOURS:.0f} hours")
    print(f"  Remaining after coarse: {config.EFFECTIVE_GPU_HOURS - total_coarse_hours:.1f} hours")
    feasible = total_coarse_hours < config.EFFECTIVE_GPU_HOURS
    print(f"  Feasible: {feasible}")

    # Save Stage 1 artifacts
    print("\n--- Saving Stage 1 artifacts ---")
    artifacts = {
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "benchmark_tasks": list(task_counts.keys()),
        "task_item_counts": task_counts,
        "total_items": full_results['total_items'],
        "baseline_accuracy": full_results['accuracy'],
        "baseline_accuracy_pct": full_results['accuracy_pct'],
        "correct_count": full_results['correct_count'],
        "degenerate_count": full_results['degenerate_count'],
        "seconds_per_item": full_results['seconds_per_item'],
        "wall_clock_full_pass": full_results['wall_clock_seconds'],
        "vram_peak_gb": peak if config.DEVICE == "cuda" else None,
        "vram_headroom_gb": (total - peak) if config.DEVICE == "cuda" else None,
        "architecture": arch_info,
        "power_analysis": power,
        "budget_estimate": {
            "time_per_pass_hours": time_per_pass_hours,
            "coarse_sweep_hours": total_coarse_hours,
            "effective_budget_hours": config.EFFECTIVE_GPU_HOURS,
            "remaining_hours": config.EFFECTIVE_GPU_HOURS - total_coarse_hours,
            "feasible": feasible,
        },
        "inference_config": {
            "torch_dtype": "float16",
            "attn_implementation": "eager",
            "deterministic_algorithms": True,
            "temperature": 0,
            "evaluation_method": "log_probability",
            "few_shot": False,
        },
    }

    artifacts_path = os.path.join(config.CALIBRATION_DIR, "stage1_artifacts.json")
    with open(artifacts_path, "w") as f:
        json.dump(artifacts, f, indent=2, default=str)
    print(f"  Saved to {artifacts_path}")

    # Save per-item baseline
    baseline_path = os.path.join(config.CALIBRATION_DIR, "baseline_per_item.json")
    with open(baseline_path, "w") as f:
        json.dump(full_results['per_item'], f, indent=2, default=str)
    print(f"  Per-item baseline saved to {baseline_path}")

    # Check if in accuracy window
    acc = full_results['accuracy_pct']
    if 25 <= acc <= 45:
        print(f"\n  PASS: Accuracy {acc:.1f}% is in the 25-45% window")
    else:
        print(f"\n  WARNING: Accuracy {acc:.1f}% is outside the 25-45% window")
        if acc < 25:
            print("  Consider using a larger model or adding few-shot examples")
        else:
            print("  Consider using a smaller model or harder tasks")

    print("\n" + "=" * 60)
    print("STAGE 1 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
