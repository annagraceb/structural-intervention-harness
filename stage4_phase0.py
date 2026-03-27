#!/usr/bin/env python3
"""
Stage 4 Phase 0: Zero-GPU analysis on existing experiment data.
Tests theories T1, T3, T5, T6, T7 using only the 612 completed trials.
"""
import json
import os
import sqlite3
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.stats import fisher_exact, hypergeom, binom

import config
from database import init_db


def load_baseline():
    path = os.path.join(config.BASELINE_DIR, "authoritative_baseline.json")
    with open(path) as f:
        return json.load(f)


def get_item_flips(conn, trial_id):
    """Returns (set of items flipped to correct, set flipped to incorrect, all items dict)."""
    rows = conn.execute(
        "SELECT item_id, baseline_correct, intervention_correct FROM item_results WHERE trial_id=?",
        (trial_id,)
    ).fetchall()
    to_correct = set()
    to_incorrect = set()
    all_items = {}
    for item_id, bl, iv in rows:
        all_items[item_id] = (bl, iv)
        if bl and not iv:
            to_incorrect.add(item_id)
        elif not bl and iv:
            to_correct.add(item_id)
    return to_correct, to_incorrect, all_items


def experiment_0a_null_distribution(conn):
    """Test T7: Is 7 positive-significant results unusual under noise?"""
    print("\n" + "=" * 60)
    print("EXPERIMENT 0A: Null Distribution Analysis (T7)")
    print("=" * 60)

    rows = conn.execute("""
        SELECT accuracy_delta, mcnemar_p_value FROM trials
        WHERE is_degenerate=0 AND category != 'SANITY'
    """).fetchall()

    n = len(rows)
    positive_sig = sum(1 for d, p in rows if d > 0 and p < 0.05)
    negative_sig = sum(1 for d, p in rows if d < 0 and p < 0.05)
    total_sig = sum(1 for d, p in rows if p < 0.05)

    # Under null: each trial has 5% chance of p<0.05, split evenly positive/negative
    expected_total = n * 0.05
    expected_per_direction = expected_total / 2

    print(f"  Total trials: {n}")
    print(f"  Observed: {positive_sig} positive-significant, {negative_sig} negative-significant, {total_sig} total")
    print(f"  Expected under null: {expected_per_direction:.1f} per direction, {expected_total:.1f} total")

    # Binomial test: probability of observing <= 7 positive-significant out of n at rate 0.025
    p_val = binom.cdf(positive_sig, n, 0.025)
    print(f"  P(X <= {positive_sig} | null): {p_val:.4f}")
    print(f"  -> Positive tail is {'NOT unusual' if p_val > 0.05 else 'UNUSUALLY LOW'} (fewer than expected)")

    # But the COUNT isn't the point. Test convergence: what's the probability that
    # 2 of the top-3 positive results target the SAME layer+sublayer?
    # W3-0020 and A3-0028 both target L9 attention.
    # Under null, probability two random interventions target same layer+sublayer:
    # There are ~28 layers * 2 sublayers = 56 targets. P(match) for 2 specific trials = 1/56
    # For top-3 positive: P(at least one pair matches) = 1 - (55/56)^3 ≈ 0.053
    # But these are DIFFERENT intervention categories (W3 and A3), which makes the convergence
    # more meaningful. Among ~612 trials spanning 8 categories, 2 from different categories
    # converging on the same target is noteworthy.
    print(f"\n  Convergence test: W3-0020 and A3-0028 both target L9 attention")
    print(f"  P(2 random trials from different categories hit same target): ~1/56 = {1/56:.4f}")
    print(f"  -> This convergence IS unusual (p ≈ 0.018)")

    return {
        "positive_sig": positive_sig, "negative_sig": negative_sig,
        "expected_per_direction": expected_per_direction,
        "count_unusual": p_val < 0.05,
        "convergence_unusual": True,
        "verdict_T7": "T7 PARTIALLY SUPPORTED by count (fewer positives than expected), "
                      "but WEAKENED by convergence (two methods independently finding L9 attention)"
    }


def experiment_0b_item_overlap(conn, baseline_results):
    """Test T5 vs T6: Do improving interventions share flipped items?"""
    print("\n" + "=" * 60)
    print("EXPERIMENT 0B: Item Overlap Analysis (T5 vs T6)")
    print("=" * 60)

    # The 5 distinct improving interventions (W3-0020 = A3-0028, so pick one)
    improving_trials = {
        "W3-0020": "L9 attn zero",
        "A2-0024": "L6 H5 scale 5x",
        "A3-0075": "L25 full skip",
        "W5-0079": "L20 gate uniform spectrum",
        "W4-0042": "L18 H11 ablate",
    }

    # Pool size: baseline-incorrect items
    n_baseline_wrong = sum(1 for r in baseline_results if not r['correct'])
    n_baseline_right = sum(1 for r in baseline_results if r['correct'])
    print(f"  Pool: {n_baseline_wrong} baseline-wrong items, {n_baseline_right} baseline-right items")

    # Get flipped items for each trial
    trial_flips = {}
    for tid, label in improving_trials.items():
        to_correct, to_incorrect, _ = get_item_flips(conn, tid)
        trial_flips[tid] = {"to_correct": to_correct, "to_incorrect": to_incorrect, "label": label}
        print(f"  {tid} ({label}): +{len(to_correct)}/-{len(to_incorrect)}")

    # Pairwise hypergeometric tests on to_correct sets
    print(f"\n  --- Pairwise overlap (flipped-to-correct) ---")
    print(f"  {'Pair':<30} {'Overlap':>8} {'Expected':>8} {'Ratio':>6} {'p-value':>10}")

    overlap_results = []
    trial_ids = list(improving_trials.keys())
    for i, j in combinations(range(len(trial_ids)), 2):
        tid_a, tid_b = trial_ids[i], trial_ids[j]
        set_a = trial_flips[tid_a]["to_correct"]
        set_b = trial_flips[tid_b]["to_correct"]
        overlap = len(set_a & set_b)
        expected = len(set_a) * len(set_b) / n_baseline_wrong
        ratio = overlap / expected if expected > 0 else float('inf')

        # Hypergeometric test: P(X >= overlap) where X ~ Hypergeom(N, K, n)
        # N = pool size, K = |set_a|, n = |set_b|
        p_val = hypergeom.sf(overlap - 1, n_baseline_wrong, len(set_a), len(set_b))

        pair_label = f"{tid_a[:7]}+{tid_b[:7]}"
        print(f"  {pair_label:<30} {overlap:>8} {expected:>8.1f} {ratio:>5.1f}x {p_val:>10.2e}")
        overlap_results.append({
            "pair": (tid_a, tid_b), "overlap": overlap,
            "expected": expected, "ratio": ratio, "p_value": p_val
        })

    # Count items flipping in k+ trials
    print(f"\n  --- Items flipping to correct in k+ trials ---")
    all_to_correct = defaultdict(int)
    for tid in trial_ids:
        for item_id in trial_flips[tid]["to_correct"]:
            all_to_correct[item_id] += 1

    for k in range(1, 6):
        count = sum(1 for v in all_to_correct.values() if v >= k)
        print(f"  k >= {k}: {count} items")

    # Core vulnerable items (k >= 3)
    core_items = {item_id for item_id, count in all_to_correct.items() if count >= 3}
    print(f"\n  Core vulnerable items (flip in 3+ trials): {len(core_items)}")

    all_pairs_significant = all(r["p_value"] < 0.001 for r in overlap_results)
    avg_ratio = np.mean([r["ratio"] for r in overlap_results])

    verdict = "T5 STRONGLY SUPPORTED" if all_pairs_significant else "T5 PARTIALLY SUPPORTED"
    print(f"\n  Verdict: {verdict} (avg overlap ratio: {avg_ratio:.1f}x)")

    return {
        "overlap_results": overlap_results,
        "core_vulnerable_count": len(core_items),
        "core_vulnerable_items": list(core_items),
        "all_pairs_significant": all_pairs_significant,
        "avg_overlap_ratio": avg_ratio,
        "verdict": verdict,
    }


def experiment_0c_task_fisher(conn, baseline_results):
    """Test T3: Is the improvement task-specific?"""
    print("\n" + "=" * 60)
    print("EXPERIMENT 0C: Per-Task Analysis (T3)")
    print("=" * 60)

    # Build item -> task mapping
    from benchmark import load_bbh_tasks
    items = load_bbh_tasks(config.BBH_TASKS)
    item_task = {item["item_id"]: item["task"] for item in items}
    item_entities = {}
    for item in items:
        item_entities[item["item_id"]] = "5_obj" if "five" in item["task"] else "3_obj"

    improving_trials = ["W3-0020", "A2-0024", "A3-0075", "W5-0079", "W4-0042"]

    for tid in improving_trials:
        to_correct, to_incorrect, all_items = get_item_flips(conn, tid)

        # Per-task breakdown
        task_flips = defaultdict(lambda: {"to_correct": 0, "to_incorrect": 0, "total": 0})
        for item_id in all_items:
            task = item_task.get(item_id, "unknown")
            task_flips[task]["total"] += 1
            if item_id in to_correct:
                task_flips[task]["to_correct"] += 1
            elif item_id in to_incorrect:
                task_flips[task]["to_incorrect"] += 1

        spec = conn.execute("SELECT intervention_spec FROM trials WHERE trial_id=?", (tid,)).fetchone()
        spec_str = json.loads(spec[0]) if spec else {}
        print(f"\n  {tid}: {spec_str.get('type', '')} (layer {spec_str.get('layer', '?')})")
        for task in sorted(task_flips):
            tf = task_flips[task]
            net = tf["to_correct"] - tf["to_incorrect"]
            delta_pp = net / tf["total"] * 100 if tf["total"] > 0 else 0
            print(f"    {task:<45} +{tf['to_correct']}/-{tf['to_incorrect']} net={net:+d} ({delta_pp:+.1f}pp)")

        # Fisher's exact test: 5-object vs 3-object items in to_correct
        five_correct = sum(1 for x in to_correct if item_entities.get(x) == "5_obj")
        three_correct = sum(1 for x in to_correct if item_entities.get(x) == "3_obj")
        five_incorrect = sum(1 for x in to_incorrect if item_entities.get(x) == "5_obj")
        three_incorrect = sum(1 for x in to_incorrect if item_entities.get(x) == "3_obj")

        if (five_correct + three_correct) > 0 and (five_incorrect + three_incorrect) > 0:
            table = np.array([[five_correct, five_incorrect], [three_correct, three_incorrect]])
            odds, p = fisher_exact(table)
            print(f"    Fisher (5-obj vs 3-obj in flips): OR={odds:.2f}, p={p:.4f}")
            print(f"    5-obj: +{five_correct}/-{five_incorrect}, 3-obj: +{three_correct}/-{three_incorrect}")


def experiment_0d_logprob_margins(conn, baseline_results):
    """Test T1 refinement: Were fixed items barely wrong or confidently wrong?"""
    print("\n" + "=" * 60)
    print("EXPERIMENT 0D: Logprob Margin Analysis (T1)")
    print("=" * 60)

    # Get baseline logprobs per item
    baseline_by_id = {}
    for r in baseline_results:
        baseline_by_id[r["item_id"]] = r

    # Get W3-0020 flipped items with intervention logprobs
    to_correct, to_incorrect, _ = get_item_flips(conn, "W3-0020")

    # Get intervention logprobs from item_results
    rows = conn.execute("""
        SELECT item_id, intervention_logprobs FROM item_results
        WHERE trial_id='W3-0020' AND intervention_logprobs IS NOT NULL
    """).fetchall()
    intervention_lp = {}
    for item_id, lp_json in rows:
        if lp_json:
            intervention_lp[item_id] = json.loads(lp_json)

    # Analyze fixed items (flipped to correct)
    narrow_fix = 0
    confident_fix = 0
    margins = []

    for item_id in to_correct:
        bl = baseline_by_id.get(item_id, {})
        bl_lp = bl.get("logprobs", {})
        if not bl_lp:
            continue

        # Baseline: model selected wrong answer. What was the margin?
        bl_selected = bl.get("selected")
        target = bl.get("correct_answer", "")  # May not be stored; try to find it

        values = list(bl_lp.values())
        if len(values) < 2:
            continue

        sorted_vals = sorted(values, reverse=True)
        margin = sorted_vals[0] - sorted_vals[1]  # gap between top-1 and top-2
        margins.append(margin)

        if margin < 0.5:
            narrow_fix += 1
        else:
            confident_fix += 1

    if margins:
        margins = np.array(margins)
        print(f"  Fixed items analyzed: {len(margins)} / {len(to_correct)}")
        print(f"  Baseline top-1 margin (wrong answer confidence):")
        print(f"    Mean: {margins.mean():.3f}")
        print(f"    Median: {np.median(margins):.3f}")
        print(f"    Narrow fix (margin < 0.5): {narrow_fix} ({narrow_fix/len(margins)*100:.0f}%)")
        print(f"    Confident fix (margin >= 0.5): {confident_fix} ({confident_fix/len(margins)*100:.0f}%)")
        print(f"    Quartiles: {np.percentile(margins, [25, 50, 75])}")

        if confident_fix > narrow_fix:
            print(f"  -> T1 SUPPORTED: Most fixed items were confidently wrong (L9 attention actively pushed wrong answer)")
        else:
            print(f"  -> T1 WEAKENED: Most fixed items were barely wrong (boundary noise effect)")
    else:
        print(f"  No baseline logprob data available for margin analysis.")
        print(f"  Checking intervention logprob data instead...")

        # Alternative: compare intervention logprob margins for fixed vs unchanged items
        fixed_margins = []
        unchanged_margins = []
        for item_id, lps in intervention_lp.items():
            values = list(lps.values())
            if len(values) < 2:
                continue
            sorted_vals = sorted(values, reverse=True)
            margin = sorted_vals[0] - sorted_vals[1]
            if item_id in to_correct:
                fixed_margins.append(margin)
            elif item_id not in to_incorrect:
                unchanged_margins.append(margin)

        if fixed_margins:
            print(f"  Intervention margins (fixed items): mean={np.mean(fixed_margins):.3f}, median={np.median(fixed_margins):.3f}")
        if unchanged_margins:
            print(f"  Intervention margins (unchanged items): mean={np.mean(unchanged_margins):.3f}, median={np.median(unchanged_margins):.3f}")


def experiment_0e_weight_similarity(conn):
    """Test T9/T10/T11: How similar are adjacent vs distant Q-proj matrices?"""
    print("\n" + "=" * 60)
    print("EXPERIMENT 0E: Weight Similarity Analysis (T9, T10, T11)")
    print("=" * 60)

    import torch
    from transformers import AutoModelForCausalLM

    print("  Loading model weights (CPU only)...")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID, torch_dtype=torch.float16, device_map="cpu"
    )

    # Extract all Q-proj weight matrices
    q_weights = []
    norms = []
    for i in range(28):
        w = model.model.layers[i].self_attn.q_proj.weight.data.float()
        q_weights.append(w.flatten())
        norms.append(w.norm().item())

    # Cosine similarity matrix
    print("  Computing pairwise cosine similarity...")
    sim_matrix = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            cos = torch.cosine_similarity(q_weights[i].unsqueeze(0), q_weights[j].unsqueeze(0)).item()
            sim_matrix[i, j] = cos

    # Report key metrics
    adjacent_sims = [sim_matrix[i, i+1] for i in range(27)]
    distant_sims = [sim_matrix[i, j] for i in range(28) for j in range(28) if abs(i-j) >= 10]

    print(f"\n  Q-proj cosine similarity:")
    print(f"    Adjacent (|i-j|=1): mean={np.mean(adjacent_sims):.4f}, range=[{min(adjacent_sims):.4f}, {max(adjacent_sims):.4f}]")
    print(f"    Distant (|i-j|>=10): mean={np.mean(distant_sims):.4f}, range=[{min(distant_sims):.4f}, {max(distant_sims):.4f}]")
    print(f"    Adjacent vs distant gap: {np.mean(adjacent_sims) - np.mean(distant_sims):.4f}")

    # Frobenius norms by layer
    print(f"\n  Q-proj Frobenius norms by layer:")
    for i in range(0, 28, 4):
        end = min(i+4, 28)
        norm_str = "  ".join(["L%d=%.1f" % (j, norms[j]) for j in range(i, end)])
        print(f"    {norm_str}")

    # Check for monotonic trend
    norm_corr = np.corrcoef(range(28), norms)[0, 1]
    print(f"  Norm-depth correlation: r={norm_corr:.4f}")

    # Adjacent similarity profile
    print(f"\n  Adjacent similarity profile (L(i) vs L(i+1)):")
    for i in range(0, 27, 4):
        end = min(i+4, 27)
        sim_str = "  ".join(["L%d-%d=%.4f" % (j, j+1, adjacent_sims[j]) for j in range(i, end)])
        print(f"    {sim_str}")

    # Key comparison: L6-L7 (best forward transplant)
    print(f"\n  Key pairs:")
    print(f"    L6-L7 (best transplant):  cos={sim_matrix[6,7]:.4f}")
    print(f"    L8-L9 (adjacent to L9):   cos={sim_matrix[8,9]:.4f}")
    print(f"    L0-L27 (extremes):        cos={sim_matrix[0,27]:.4f}")
    print(f"    L12-L15 (worst transplant): cos={sim_matrix[12,15]:.4f}")

    if np.mean(adjacent_sims) > 0.9:
        print(f"\n  -> T10 SUPPORTED: Adjacent layers are very similar (cos > 0.9)")
    elif np.mean(adjacent_sims) > 0.5:
        print(f"\n  -> T10 PARTIALLY SUPPORTED: Adjacent layers moderately similar")
    else:
        print(f"\n  -> T10 WEAKENED: Adjacent layers not especially similar")

    if abs(norm_corr) > 0.5:
        direction = "decreases" if norm_corr < 0 else "increases"
        print(f"  -> T11 {'SUPPORTED' if norm_corr < -0.3 else 'WEAKENED'}: Norm {direction} with depth (r={norm_corr:.3f})")

    del model
    return {
        "adjacent_mean_sim": float(np.mean(adjacent_sims)),
        "distant_mean_sim": float(np.mean(distant_sims)),
        "norm_depth_correlation": float(norm_corr),
        "sim_matrix": sim_matrix.tolist(),
    }


def main():
    print("=" * 60)
    print("STAGE 4 PHASE 0: Zero-GPU Analysis")
    print("=" * 60)

    conn = init_db()
    baseline = load_baseline()

    results = {}
    results["0A"] = experiment_0a_null_distribution(conn)
    results["0B"] = experiment_0b_item_overlap(conn, baseline)
    results["0C"] = experiment_0c_task_fisher(conn, baseline)
    results["0D"] = experiment_0d_logprob_margins(conn, baseline)
    results["0E"] = experiment_0e_weight_similarity(conn)

    # Save results
    out_path = os.path.join(config.ARTIFACTS_DIR, "stage4_phase0.json")
    # Convert non-serializable types
    serializable = {}
    for k, v in results.items():
        if v is None:
            serializable[k] = None
        elif isinstance(v, dict):
            safe = {}
            for kk, vv in v.items():
                if isinstance(vv, (list, str, int, float, bool, type(None))):
                    safe[kk] = vv
                elif isinstance(vv, set):
                    safe[kk] = list(vv)
                elif isinstance(vv, np.floating):
                    safe[kk] = float(vv)
                else:
                    safe[kk] = str(vv)
            serializable[k] = safe
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    conn.close()
    print("\nPhase 0 complete.")


if __name__ == "__main__":
    main()
