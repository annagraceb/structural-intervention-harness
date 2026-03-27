"""
Statistical tests and tier classification per spec v0.4.
"""
import numpy as np
from scipy.stats import binom


def mcnemar_exact_test(b: int, c: int) -> float:
    """McNemar's exact test (two-sided).

    b = items that were correct at baseline but incorrect after intervention
    c = items that were incorrect at baseline but correct after intervention

    Under H0: each discordant pair is equally likely to be b-type or c-type.
    P(X <= min(b,c)) under Binomial(b+c, 0.5), two-sided.
    """
    n = b + c
    if n == 0:
        return 1.0

    k = min(b, c)
    # Two-sided p-value: P(X <= k) * 2, capped at 1.0
    p_value = 2.0 * binom.cdf(k, n, 0.5)
    return min(p_value, 1.0)


def benjamini_hochberg(p_values: list[float], fdr: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg procedure. Returns list of booleans (significant or not).

    Single global correction across all interventions per spec v0.4.
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort p-values and track original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    significant = [False] * n

    # BH threshold for rank i (1-indexed): fdr * i / n
    for rank_0, (orig_idx, p) in enumerate(indexed):
        rank = rank_0 + 1
        threshold = fdr * rank / n
        if p <= threshold:
            significant[orig_idx] = True
        else:
            # Once we find one that doesn't pass, the rest also won't
            # (this is the step-up procedure)
            break

    # Actually BH step-up: find the largest rank i where p(i) <= fdr*i/n
    # then reject all hypotheses with rank <= i
    max_significant_rank = 0
    for rank_0, (orig_idx, p) in enumerate(indexed):
        rank = rank_0 + 1
        threshold = fdr * rank / n
        if p <= threshold:
            max_significant_rank = rank

    significant = [False] * n
    for rank_0, (orig_idx, p) in enumerate(indexed):
        rank = rank_0 + 1
        if rank <= max_significant_rank:
            significant[orig_idx] = True

    return significant


def classify_tier(
    accuracy_delta_pp: float,
    items_flipped_to_correct: int,
    items_flipped_to_incorrect: int,
    bh_significant: bool,
    n_items: int,
    min_delta_pp: float = 3.0,
    min_directional_ratio: float = 0.70,
) -> tuple[int | None, str]:
    """Classify an intervention result into Tier 1, 2, or None.

    Returns (tier, justification).
    Tier 3 requires additional checks (pre-registered predictions) done externally.
    """
    if not bh_significant:
        return None, "Not significant after BH correction"

    # Tier 1: statistically detectable
    total_discordant = items_flipped_to_correct + items_flipped_to_incorrect
    if total_discordant == 0:
        return None, "No discordant pairs"

    # Check directional asymmetry (this should be implied by BH significance, but be explicit)
    dominant_direction = max(items_flipped_to_correct, items_flipped_to_incorrect)
    directional_ratio = dominant_direction / total_discordant if total_discordant > 0 else 0

    # Tier 2 check
    if abs(accuracy_delta_pp) >= min_delta_pp and directional_ratio >= min_directional_ratio:
        if accuracy_delta_pp > 0:
            tier2_type = "Tier 2-constructive"
        else:
            tier2_type = "Tier 2-destructive"
        return 2, f"{tier2_type}: delta={accuracy_delta_pp:+.1f}pp, directional_ratio={directional_ratio:.2f}, flipped_correct={items_flipped_to_correct}, flipped_incorrect={items_flipped_to_incorrect}"

    return 1, f"Tier 1: delta={accuracy_delta_pp:+.1f}pp, directional_ratio={directional_ratio:.2f}, flipped_correct={items_flipped_to_correct}, flipped_incorrect={items_flipped_to_incorrect}"
