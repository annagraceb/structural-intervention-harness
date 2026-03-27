#!/usr/bin/env python3
"""Re-run COMB-04 and COMB-05 after ActivationHook bug fix."""
import json
import os
import time

import torch

import config
config.configure_determinism()

from benchmark import load_bbh_tasks, load_model_and_tokenizer, evaluate_item_logprob
from weight_manager import WeightManager
from interventions import ActivationHook
from trial_runner import run_trial, print_trial_summary
from database import init_db, get_completed_trial_ids
from stage3_run import execute_intervention
from stage4_phase1 import execute_combined_intervention


def main():
    print("Re-running COMB-04 and COMB-05 (ActivationHook fix)")

    items = load_bbh_tasks(config.BBH_TASKS)
    model, tokenizer = load_model_and_tokenizer(config.MODEL_ID, config.DEVICE)
    wm = WeightManager(model)
    db_conn = init_db()

    baseline_path = os.path.join(config.BASELINE_DIR, "authoritative_baseline.json")
    with open(baseline_path) as f:
        baseline_results = json.load(f)

    trials = [
        {
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
        },
        {
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
        },
    ]

    completed = get_completed_trial_ids(db_conn)

    for trial_spec in trials:
        trial_id = trial_spec['trial_id']
        spec = trial_spec['intervention_spec']

        if trial_id in completed:
            print(f"\n  {trial_id}: already completed, skipping")
            continue

        print(f"\n  Running {trial_id}: {spec['description']}")
        try:
            weight_mods, hook = execute_combined_intervention(model, wm, spec, config.DEVICE)

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

    db_conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
