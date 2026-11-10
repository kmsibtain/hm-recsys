"""Evaluation metrics for recommendation systems.

Implements Average Precision at K (AP@K) and Mean Average Precision at K (MAP@K)
matching the Kaggle H&M Personalized Fashion Recommendations competition definition.
"""

from collections.abc import Iterable, Sequence
from typing import Any


def apk(actual: Iterable[Any], predicted: Sequence[Any], k: int = 12) -> float:
    """Computes Average Precision at K (AP@K) for a single customer.

    Formula:
        AP@K = sum_{i=1}^{min(len(predicted), k)} (P(i) * rel(i)) / min(len(actual), k)

    where:
        - P(i) is precision at cut-off i: (hits in predicted[:i]) / i
        - rel(i) is 1 if predicted[i-1] in actual, 0 otherwise
        - Only the first occurrence of each relevant item in predicted counts as a hit
        - Denominator is min(len(set(actual)), k)

    Args:
        actual: Collection of ground-truth item IDs purchased by the customer.
        predicted: Ordered sequence of recommended item IDs (up to k items).
        k: Maximum number of recommendations to evaluate (default: 12).

    Returns:
        Average precision score in range [0.0, 1.0]. Returns 0.0 if actual or predicted is empty.
    """
    if actual is None or len(actual) == 0:
        return 0.0

    actual_set = set(actual)
    n_actual = len(actual_set)
    if n_actual == 0:
        return 0.0

    if predicted is None or len(predicted) == 0:
        return 0.0

    score = 0.0
    num_hits = 0.0
    seen_predicted = set()

    for i, item in enumerate(predicted[:k], start=1):
        if item in actual_set and item not in seen_predicted:
            num_hits += 1.0
            score += num_hits / i
            seen_predicted.add(item)

    denominator = min(n_actual, k)
    return score / denominator


def mapk(
    actuals: Sequence[Iterable[Any]],
    predictions: Sequence[Sequence[Any]],
    k: int = 12,
    filter_empty_actuals: bool = True,
) -> float:
    """Computes Mean Average Precision at K (MAP@K) across multiple customers.

    Args:
        actuals: Sequence of ground-truth items per customer.
        predictions: Sequence of recommended item lists per customer.
        k: Cutoff rank (default: 12).
        filter_empty_actuals: If True, only evaluate customers who have at least
            one purchase in the target period (standard competition evaluation choice).
            If False, includes customers with empty actuals (yielding 0.0 AP for them).

    Returns:
        Mean Average Precision score in range [0.0, 1.0].
    """
    if len(actuals) != len(predictions):
        raise ValueError(
            f"Length mismatch: {len(actuals)} actual lists vs {len(predictions)} prediction lists"
        )

    scores = []
    for actual, pred in zip(actuals, predictions):
        if filter_empty_actuals and (actual is None or len(actual) == 0):
            continue
        scores.append(apk(actual, pred, k=k))

    if not scores:
        return 0.0

    return sum(scores) / len(scores)
