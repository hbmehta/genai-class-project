"""Compute inter-rater agreement metrics between human and model-as-judge scores."""

from collections import Counter


def percent_agreement(scores_a: list[int], scores_b: list[int]) -> float:
    """Fraction of scores where both raters agree exactly."""
    if not scores_a or len(scores_a) != len(scores_b):
        raise ValueError("Score lists must be non-empty and equal length")
    matches = sum(a == b for a, b in zip(scores_a, scores_b))
    return matches / len(scores_a)


def cohen_kappa(
    scores_a: list[int],
    scores_b: list[int],
    n_categories: int = 3,
    categories: list[int] | None = None,
) -> float:
    """
    Compute Cohen's kappa for two raters with ordinal scores.

    kappa = (p_o - p_e) / (1 - p_e)

    Args:
        scores_a: Human rater scores.
        scores_b: Model-as-judge scores.
        n_categories: Number of rating categories (default 3: scores 1, 2, 3).
        categories: Explicit list of category values (overrides n_categories).

    Returns:
        kappa float. Interpretation:
            < 0.20  = slight
            0.20–0.40 = fair
            0.40–0.60 = moderate
            0.60–0.80 = substantial
            > 0.80  = near-perfect
    """
    if not scores_a or len(scores_a) != len(scores_b):
        raise ValueError("Score lists must be non-empty and equal length")

    if categories is None:
        categories = list(range(1, n_categories + 1))

    n = len(scores_a)

    # Observed agreement
    p_o = sum(a == b for a, b in zip(scores_a, scores_b)) / n

    # Expected agreement
    count_a = Counter(scores_a)
    count_b = Counter(scores_b)
    p_e = sum(
        (count_a.get(cat, 0) / n) * (count_b.get(cat, 0) / n)
        for cat in categories
    )

    if p_e == 1.0:
        return 1.0

    return (p_o - p_e) / (1.0 - p_e)


def interpret_kappa(kappa: float) -> str:
    if kappa < 0.0:
        return "less than chance"
    elif kappa < 0.20:
        return "slight"
    elif kappa < 0.40:
        return "fair"
    elif kappa < 0.60:
        return "moderate"
    elif kappa < 0.80:
        return "substantial"
    else:
        return "near-perfect"


def print_agreement_report(
    human_scores: list[int],
    model_scores: list[int],
    dimension: str = "relevance",
) -> None:
    pa = percent_agreement(human_scores, model_scores)
    k = cohen_kappa(human_scores, model_scores)
    interp = interpret_kappa(k)
    print(f"\n[{dimension}]")
    print(f"  Percent agreement : {pa:.1%}")
    print(f"  Cohen's kappa     : {k:.3f} ({interp})")
