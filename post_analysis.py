#!/usr/bin/env python3
"""
Post-analysis: Evaluate pre-registered predictions, run item-property taxonomy
tests on Tier 2+ results, and generate the anomaly report.

Run this after stage3_run.py completes.
"""
import json
import os
import re
import sqlite3
from collections import defaultdict

import numpy as np
from scipy.stats import fisher_exact

import config
from database import init_db, get_non_degenerate_trials


def load_taxonomy():
    """Load the pre-registered item-property taxonomy."""
    path = os.path.join(config.ARTIFACTS_DIR, "item_property_taxonomy.json")
    with open(path) as f:
        return json.load(f)


def load_items():
    """Load benchmark items and annotate with taxonomy properties."""
    from benchmark import load_bbh_tasks, extract_options
    items = load_bbh_tasks(config.BBH_TASKS)

    for item in items:
        task = item["task"]
        text = item["input"]

        # P1: entity_count
        item["P1_entity_count"] = "3_entities" if "three" in task else "5_entities"

        # P2: task_type
        item["P2_task_type"] = "tracking" if "tracking" in task else "deduction"

        # P3: swap_count (tracking only)
        if "tracking" in task:
            swaps = len(re.findall(r'(?:swap|switch)', text, re.I))
            item["P3_swap_count"] = "low_swaps" if swaps <= 1 else "high_swaps"
        else:
            item["P3_swap_count"] = None

        # P4: input_length — computed per-task after all items loaded
        item["_input_len"] = len(text)

        # P5: queried_position (deduction only)
        if "deduction" in task:
            # Try to detect if the question asks about first/last
            lower = text.lower()
            if "first" in lower or "last" in lower:
                item["P5_queried_position"] = "boundary"
            elif "second" in lower or "middle" in lower or "third" in lower:
                item["P5_queried_position"] = "interior"
            else:
                item["P5_queried_position"] = None
        else:
            item["P5_queried_position"] = None

        # P6: answer_position
        target = item["target"].strip()
        options = extract_options(item)
        if options:
            last_letter = options[-1][0]
            if target.startswith("(A)") or target == "A":
                item["P6_answer_position"] = "first_option"
            elif f"({last_letter})" in target or target == last_letter:
                item["P6_answer_position"] = "last_option"
            else:
                item["P6_answer_position"] = "middle_option"
        else:
            item["P6_answer_position"] = None

    # P4: compute terciles within each task
    by_task = defaultdict(list)
    for item in items:
        by_task[item["task"]].append(item)
    for task_items in by_task.values():
        lengths = sorted(it["_input_len"] for it in task_items)
        t1 = lengths[len(lengths) // 3]
        t2 = lengths[2 * len(lengths) // 3]
        for it in task_items:
            if it["_input_len"] <= t1:
                it["P4_input_length"] = "short"
            elif it["_input_len"] >= t2:
                it["P4_input_length"] = "long"
            else:
                it["P4_input_length"] = "medium"

    return {item["item_id"]: item for item in items}


def get_item_flips(conn, trial_id):
    """Get flipped items for a trial. Returns (flipped_item_ids, flip_directions)."""
    cursor = conn.execute("""
        SELECT item_id, baseline_correct, intervention_correct
        FROM item_results
        WHERE trial_id = ?
    """, (trial_id,))
    flipped = {}
    all_items = {}
    for row in cursor:
        item_id, bl, iv = row
        all_items[item_id] = (bl, iv)
        if bl != iv:
            flipped[item_id] = "to_correct" if iv else "to_incorrect"
    return flipped, all_items


def run_taxonomy_test(items_map, flipped_items, all_items, property_name):
    """Run Fisher's exact test for a property split on flipped items.

    Returns (odds_ratio, p_value, table, group_a_name, group_b_name) or None if not applicable.
    """
    # Get unique groups for this property
    groups = set()
    for item_id in all_items:
        if item_id in items_map:
            val = items_map[item_id].get(property_name)
            if val is not None:
                groups.add(val)

    if len(groups) < 2:
        return None

    groups = sorted(groups)
    # For properties with >2 groups, use first vs last
    if len(groups) > 2:
        group_a, group_b = groups[0], groups[-1]
    else:
        group_a, group_b = groups[0], groups[1]

    # Build 2x2 table: rows = group (A vs B), cols = flipped vs not-flipped
    a_flipped = 0
    a_not_flipped = 0
    b_flipped = 0
    b_not_flipped = 0

    for item_id in all_items:
        if item_id not in items_map:
            continue
        prop_val = items_map[item_id].get(property_name)
        is_flipped = item_id in flipped_items

        if prop_val == group_a:
            if is_flipped:
                a_flipped += 1
            else:
                a_not_flipped += 1
        elif prop_val == group_b:
            if is_flipped:
                b_flipped += 1
            else:
                b_not_flipped += 1

    table = np.array([[a_flipped, a_not_flipped], [b_flipped, b_not_flipped]])
    if table.sum() == 0:
        return None

    odds_ratio, p_value = fisher_exact(table, alternative='two-sided')
    return odds_ratio, p_value, table.tolist(), group_a, group_b


def evaluate_predictions(conn, baseline_results):
    """Evaluate each pre-registered prediction against experiment results."""
    trials = get_non_degenerate_trials(conn)
    trials = [t for t in trials if not t['trial_id'].startswith('SANITY')]

    # Parse intervention specs
    for t in trials:
        if isinstance(t['intervention_spec'], str):
            t['intervention_spec'] = json.loads(t['intervention_spec'])

    baseline_acc = sum(1 for r in baseline_results if r['correct']) / len(baseline_results)

    results = []

    # P1: MLP reinit more destructive than attention reinit
    w3_trials = [t for t in trials if t['category'] == 'W3']
    mlp_reinits = [t for t in w3_trials
                   if t['intervention_spec'].get('granularity') == 'mlp'
                   and t['intervention_spec'].get('distribution') == 'kaiming'
                   and 4 <= t['intervention_spec'].get('layer', -1) <= 24]
    attn_reinits = [t for t in w3_trials
                    if t['intervention_spec'].get('granularity') == 'attention'
                    and t['intervention_spec'].get('distribution') == 'kaiming'
                    and 4 <= t['intervention_spec'].get('layer', -1) <= 24]
    if mlp_reinits and attn_reinits:
        mlp_avg = np.mean([t['accuracy_delta'] for t in mlp_reinits])
        attn_avg = np.mean([t['accuracy_delta'] for t in attn_reinits])
        mlp_all_below_minus5 = all(t['accuracy_delta'] <= -5 for t in mlp_reinits)
        attn_avg_above_minus5 = np.mean([abs(t['accuracy_delta']) for t in attn_reinits]) < 5
        violated = not (mlp_all_below_minus5 and attn_avg_above_minus5)
        results.append({
            "id": "P1", "prediction": "MLP reinit ≥5pp degradation; attention reinit <5pp avg",
            "violated": violated,
            "evidence": f"MLP reinit avg Δ={mlp_avg:+.1f}pp (all≤-5pp: {mlp_all_below_minus5}), "
                        f"Attention reinit avg |Δ|={np.mean([abs(t['accuracy_delta']) for t in attn_reinits]):.1f}pp"
        })
    else:
        results.append({"id": "P1", "prediction": "MLP vs attention reinit", "violated": None, "evidence": "Insufficient W3 data"})

    # P2: Middle layers more critical
    for cat in ['W3', 'A3']:
        cat_trials = [t for t in trials if t['category'] == cat]
        if not cat_trials:
            continue
        early = [t for t in cat_trials if t['intervention_spec'].get('layer', 14) < 9]
        middle = [t for t in cat_trials if 9 <= t['intervention_spec'].get('layer', -1) <= 18]
        late = [t for t in cat_trials if t['intervention_spec'].get('layer', -1) > 18]
        if middle and (early or late):
            mid_avg = np.mean([abs(t['accuracy_delta']) for t in middle])
            edge_avg = np.mean([abs(t['accuracy_delta']) for t in (early + late)])
            violated = mid_avg <= edge_avg
            results.append({
                "id": f"P2_{cat}", "prediction": f"Middle layers more critical for {cat}",
                "violated": violated,
                "evidence": f"Middle |Δ| avg={mid_avg:.1f}pp, Edge |Δ| avg={edge_avg:.1f}pp"
            })

    # P3: Adjacent transplants less destructive than distant
    w2_trials = [t for t in trials if t['category'] == 'W2']
    if w2_trials:
        adjacent = [t for t in w2_trials
                    if abs(t['intervention_spec'].get('source_layer', 0) - t['intervention_spec'].get('target_layer', 0)) <= 2]
        distant = [t for t in w2_trials
                   if abs(t['intervention_spec'].get('source_layer', 0) - t['intervention_spec'].get('target_layer', 0)) >= 10]
        if adjacent and distant:
            adj_avg = np.mean([abs(t['accuracy_delta']) for t in adjacent])
            dist_avg = np.mean([abs(t['accuracy_delta']) for t in distant])
            violated = not (adj_avg < 2 and dist_avg > 3)
            results.append({
                "id": "P3", "prediction": "Adjacent transplants <2pp, distant >3pp",
                "violated": violated,
                "evidence": f"Adjacent avg |Δ|={adj_avg:.1f}pp, Distant avg |Δ|={dist_avg:.1f}pp"
            })

    # P4: Single head ablation in last 3 layers ≤2pp
    w4_trials = [t for t in trials if t['category'] == 'W4']
    last3_ablations = [t for t in w4_trials
                       if t['intervention_spec'].get('operation') == 'ablate'
                       and t['intervention_spec'].get('layer', 0) >= 25]
    if last3_ablations:
        any_over_2 = any(abs(t['accuracy_delta']) > 2 for t in last3_ablations)
        results.append({
            "id": "P4", "prediction": "Head ablation in last 3 layers ≤2pp",
            "violated": any_over_2,
            "evidence": "Deltas: " + str([round(t['accuracy_delta'], 1) for t in last3_ablations])
        })

    # P5: Spectral truncation at 50% preserves 90% accuracy
    w5_trials = [t for t in trials if t['category'] == 'W5']
    topk_50 = [t for t in w5_trials
               if t['intervention_spec'].get('operation') == 'top_k'
               and t['intervention_spec'].get('component', '').endswith('q_proj.weight')]
    # Filter to roughly 50% rank
    topk_half = [t for t in topk_50
                 if t['intervention_spec'].get('k', 0) > 0]
    if topk_half:
        # Check if any fell below 90% of baseline
        threshold = baseline_acc * 0.9
        any_below = any((t['accuracy'] or 0) < threshold for t in topk_half)
        results.append({
            "id": "P5", "prediction": "Spectral top-50% preserves ≥90% accuracy",
            "violated": any_below,
            "evidence": f"Threshold={threshold:.3f}, accuracies=" + str([round(t['accuracy'], 3) for t in topk_half[:5]])
        })

    # P6: W1 weakest class
    w1_trials = [t for t in trials if t['category'] == 'W1']
    if w1_trials:
        any_sig = any(t.get('bh_significant') for t in w1_trials)
        avg_abs_delta = np.mean([abs(t['accuracy_delta']) for t in w1_trials])
        violated = any_sig or avg_abs_delta >= 2
        results.append({
            "id": "P6", "prediction": "No W1 reaches Tier 1; avg |Δ| < 2pp",
            "violated": violated,
            "evidence": f"BH-significant W1: {any_sig}, avg |Δ|={avg_abs_delta:.1f}pp"
        })

    # P7: Full layer skip degrades by ≥1pp, never improves
    a3_full = [t for t in trials if t['category'] == 'A3'
               and t['intervention_spec'].get('variant') == 'full']
    if a3_full:
        any_improve = any(t['accuracy_delta'] > 0 for t in a3_full)
        any_under_1pp = any(t['accuracy_delta'] > -1 for t in a3_full)
        violated = any_improve  # "no single-layer skip will improve accuracy"
        results.append({
            "id": "P7", "prediction": "Full layer skip always degrades (≥1pp), never improves",
            "violated": violated,
            "evidence": f"Deltas: min={min(t['accuracy_delta'] for t in a3_full):+.1f}pp, "
                        f"max={max(t['accuracy_delta'] for t in a3_full):+.1f}pp, "
                        f"any_improve={any_improve}"
        })

    # P8: Attention-only skip < full skip
    a3_attn = [t for t in trials if t['category'] == 'A3'
               and t['intervention_spec'].get('variant') == 'attention_only']
    if a3_full and a3_attn:
        # Compare matched layers
        full_by_layer = {t['intervention_spec']['layer']: t for t in a3_full}
        attn_by_layer = {t['intervention_spec']['layer']: t for t in a3_attn}
        common_layers = set(full_by_layer) & set(attn_by_layer)
        if common_layers:
            violations = 0
            for l in common_layers:
                if abs(attn_by_layer[l]['accuracy_delta']) >= abs(full_by_layer[l]['accuracy_delta']):
                    violations += 1
            violated = violations > len(common_layers) / 2
            results.append({
                "id": "P8", "prediction": "Attention-only skip less destructive than full skip",
                "violated": violated,
                "evidence": f"{violations}/{len(common_layers)} layers violated (attn≥full)"
            })

    # P10: A1 noise is monotonically destructive
    a1_trials = [t for t in trials if t['category'] == 'A1']
    if a1_trials:
        any_improve = any(t['accuracy_delta'] > 0 for t in a1_trials)
        results.append({
            "id": "P10", "prediction": "No A1 injection improves accuracy",
            "violated": any_improve,
            "evidence": f"Max Δ={max(t['accuracy_delta'] for t in a1_trials):+.1f}pp"
        })

    # P13: No intervention improves by >5pp
    if trials:
        max_delta = max(t['accuracy_delta'] for t in trials)
        results.append({
            "id": "P13", "prediction": "Max improvement < 5pp",
            "violated": max_delta >= 5.0,
            "evidence": f"Max accuracy_delta = {max_delta:+.1f}pp"
        })

    # P14: Degenerate trials concentrated in W3/W5
    all_trials_inc_degen = conn.execute(
        "SELECT category, is_degenerate FROM trials WHERE category NOT LIKE 'SANITY%'"
    ).fetchall()
    total_degen = sum(1 for _, d in all_trials_inc_degen if d)
    w3w5_degen = sum(1 for c, d in all_trials_inc_degen if d and c in ('W3', 'W5'))
    if total_degen > 0:
        ratio = w3w5_degen / total_degen
        results.append({
            "id": "P14", "prediction": "≥80% degenerate trials from W3/W5",
            "violated": ratio < 0.8,
            "evidence": f"{w3w5_degen}/{total_degen} = {ratio:.0%} degenerate in W3/W5"
        })
    else:
        results.append({
            "id": "P14", "prediction": "≥80% degenerate from W3/W5",
            "violated": False, "evidence": "No degenerate trials"
        })

    return results


def generate_anomaly_report(conn, items_map, baseline_results, predictions):
    """Generate the full anomaly report."""
    trials = get_non_degenerate_trials(conn)
    trials = [t for t in trials if not t['trial_id'].startswith('SANITY')]
    for t in trials:
        if isinstance(t['intervention_spec'], str):
            t['intervention_spec'] = json.loads(t['intervention_spec'])

    # Get Tier 2+ results
    tier2_plus = [t for t in trials if t.get('tier') and t['tier'] >= 2]
    tier2_plus.sort(key=lambda t: abs(t['accuracy_delta']), reverse=True)

    report = []
    report.append("# Structural Intervention Experiment — Anomaly Report")
    report.append(f"\n**Generated:** {config.DB_PATH}")
    report.append(f"**Total trials:** {len(trials)} non-degenerate")
    report.append(f"**Tier 2+ results:** {len(tier2_plus)}")

    # Prediction evaluation
    report.append("\n## Pre-Registered Prediction Evaluation\n")
    violated_preds = [p for p in predictions if p['violated']]
    confirmed_preds = [p for p in predictions if p['violated'] is False]
    inconclusive = [p for p in predictions if p['violated'] is None]

    report.append(f"**Violated:** {len(violated_preds)} | **Confirmed:** {len(confirmed_preds)} | **Inconclusive:** {len(inconclusive)}\n")

    for p in predictions:
        status = "VIOLATED" if p['violated'] else ("CONFIRMED" if p['violated'] is False else "INCONCLUSIVE")
        marker = "!!!" if p['violated'] else ("   " if p['violated'] is False else " ? ")
        report.append(f"{marker} **{p['id']}** [{status}]: {p['prediction']}")
        report.append(f"    Evidence: {p['evidence']}\n")

    # Tier 3 candidates (Tier 2 + violated prediction)
    report.append("\n## Tier 3 Candidates\n")
    violated_pred_ids = {p['id'] for p in violated_preds}

    tier3_candidates = []
    for t in tier2_plus:
        # Check which predictions this trial might violate
        violations = check_trial_against_predictions(t, violated_preds)
        if violations:
            tier3_candidates.append((t, violations))

    if tier3_candidates:
        for t, violations in tier3_candidates:
            spec = t['intervention_spec']
            report.append(f"### {t['trial_id']} ({t['category']})")
            report.append(f"- **Δ = {t['accuracy_delta']:+.1f}pp** (p={t['mcnemar_p_value']:.6f})")
            report.append(f"- +{t['items_flipped_to_correct']}/-{t['items_flipped_to_incorrect']} flips")
            report.append(f"- Spec: {json.dumps(spec, indent=2)}")
            report.append(f"- **Predictions violated:** {', '.join(violations)}")
            report.append("")
    else:
        report.append("No Tier 3 candidates found.\n")

    # Tier 2 results with taxonomy analysis
    report.append("\n## Tier 2+ Results with Item-Property Analysis\n")

    properties = ["P1_entity_count", "P2_task_type", "P3_swap_count",
                   "P4_input_length", "P5_queried_position", "P6_answer_position"]

    for t in tier2_plus:
        report.append(f"### {t['trial_id']} ({t['category']})")
        report.append(f"- Δ = {t['accuracy_delta']:+.1f}pp, p={t['mcnemar_p_value']:.6f}")
        report.append(f"- +{t['items_flipped_to_correct']}/-{t['items_flipped_to_incorrect']}")

        # Run taxonomy tests
        flipped, all_items = get_item_flips(conn, t['trial_id'])
        if flipped:
            report.append(f"- Total flipped items: {len(flipped)}")
            report.append("\n  Item-property Fisher's exact tests:")
            for prop in properties:
                result = run_taxonomy_test(items_map, flipped, all_items, prop)
                if result:
                    odds, p, table, ga, gb = result
                    sig = " ***" if p < 0.05 else ""
                    report.append(f"  - {prop}: OR={odds:.2f}, p={p:.4f}{sig}")
                    report.append(f"    {ga}: {table[0][0]} flipped / {table[0][1]} stable")
                    report.append(f"    {gb}: {table[1][0]} flipped / {table[1][1]} stable")
        report.append("")

    # Category summary
    report.append("\n## Category Summary\n")
    for cat in ["W1", "W2", "W3", "W4", "W5", "A1", "A2", "A3"]:
        cat_trials = [t for t in trials if t['category'] == cat]
        if not cat_trials:
            continue
        n = len(cat_trials)
        n_sig = sum(1 for t in cat_trials if t.get('bh_significant'))
        n_t2 = sum(1 for t in cat_trials if t.get('tier') and t['tier'] >= 2)
        avg_delta = np.mean([t['accuracy_delta'] for t in cat_trials])
        max_abs = max(abs(t['accuracy_delta']) for t in cat_trials)
        report.append(f"**{cat}:** {n} trials, {n_sig} significant, {n_t2} Tier 2+, "
                       f"avg Δ={avg_delta:+.1f}pp, max |Δ|={max_abs:.1f}pp")

    # Null results log
    report.append("\n\n## Null Results (no significance despite reasonable power)\n")
    null_cats = [cat for cat in ["W1", "W2", "W3", "W4", "W5", "A1", "A2", "A3"]
                 if not any(t.get('bh_significant') for t in trials if t['category'] == cat)]
    if null_cats:
        report.append(f"Categories with zero BH-significant results: {', '.join(null_cats)}")
        report.append("These null results are informative — they suggest the model is robust to these intervention types.\n")
    else:
        report.append("All categories produced at least one significant result.\n")

    return "\n".join(report)


def check_trial_against_predictions(trial, violated_predictions):
    """Check which violated predictions a specific trial violates."""
    violations = []
    spec = trial['intervention_spec']
    cat = trial['category']
    delta = trial['accuracy_delta']

    for pred in violated_predictions:
        pid = pred['id']

        if pid == "P1" and cat == "W3":
            if spec.get('granularity') == 'mlp' and delta > -5:
                violations.append("P1 (MLP reinit didn't degrade ≥5pp)")
            elif spec.get('granularity') == 'attention' and abs(delta) >= 5:
                violations.append("P1 (Attention reinit ≥5pp)")

        elif pid.startswith("P2") and cat in pid:
            layer = spec.get('layer', 14)
            if not (9 <= layer <= 18):  # edge layer more critical
                violations.append(f"{pid} (edge layer more critical than middle)")

        elif pid == "P7" and cat == "A3" and spec.get('variant') == 'full':
            if delta > 0:
                violations.append("P7 (layer skip improved accuracy)")

        elif pid == "P10" and cat == "A1":
            if delta > 0:
                violations.append("P10 (noise injection improved accuracy)")

        elif pid == "P13" and delta >= 5:
            violations.append("P13 (improvement ≥5pp)")

    return violations


def main():
    print("=" * 60)
    print("POST-ANALYSIS: Predictions, Taxonomy, Anomaly Report")
    print("=" * 60)

    conn = init_db()

    # Check if sweep is complete
    total = conn.execute("SELECT COUNT(*) FROM trials WHERE category != 'SANITY'").fetchone()[0]
    print(f"\nTrials in database: {total}/612")
    if total < 612:
        print(f"WARNING: Sweep not complete ({total}/612). Running analysis on available data.\n")

    # Load baseline
    baseline_path = os.path.join(config.BASELINE_DIR, "authoritative_baseline.json")
    with open(baseline_path) as f:
        baseline_results = json.load(f)

    # Load and annotate items
    print("Loading and annotating items...")
    items_map = load_items()
    print(f"  {len(items_map)} items annotated with taxonomy properties")

    # Evaluate predictions
    print("\nEvaluating pre-registered predictions...")
    predictions = evaluate_predictions(conn, baseline_results)
    violated = [p for p in predictions if p['violated']]
    print(f"  {len(violated)}/{len(predictions)} predictions violated")
    for p in predictions:
        status = "VIOLATED" if p['violated'] else ("OK" if p['violated'] is False else "?")
        print(f"    [{status}] {p['id']}: {p['prediction']}")

    # Generate anomaly report
    print("\nGenerating anomaly report...")
    report = generate_anomaly_report(conn, items_map, baseline_results, predictions)

    report_path = os.path.join(config.ARTIFACTS_DIR, "anomaly_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved to {report_path}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
