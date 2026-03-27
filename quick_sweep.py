#!/usr/bin/env python3
"""Quick sweep: test multiple BBH tasks on multiple models to find the 25-45% window."""
import sys
sys.path.insert(0, '.')

import config
config.configure_determinism()

from benchmark import load_bbh_tasks, run_benchmark, load_model_and_tokenizer

ALL_TASKS = [
    "tracking_shuffled_objects_three_objects",
    "tracking_shuffled_objects_five_objects",
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "navigate",
    "date_understanding",
    "web_of_lies",
    "boolean_expressions",
    "sports_understanding",
]

MODELS = [
    "Qwen/Qwen2.5-1.5B",
    # "stabilityai/stablelm-2-1_6b",  # already tested
]

for model_id in MODELS:
    print(f"\n{'='*60}")
    print(f"MODEL: {model_id}")
    print(f"{'='*60}")
    model, tokenizer = load_model_and_tokenizer(model_id, config.DEVICE)

    for task in ALL_TASKS:
        items = load_bbh_tasks([task])
        results = run_benchmark(model, tokenizer, items, config.DEVICE, max_items=50, verbose=False)
        print(f"  {task}: {results['accuracy_pct']:.1f}% ({results['correct_count']}/50, {results['seconds_per_item']:.3f}s/item)")

    # Also test a combined set
    all_items = load_bbh_tasks(ALL_TASKS)
    results = run_benchmark(model, tokenizer, all_items, config.DEVICE, max_items=200, verbose=False)
    print(f"\n  COMBINED (200 sample): {results['accuracy_pct']:.1f}%")

    del model
    import torch
    torch.cuda.empty_cache()
