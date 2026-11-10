"""Unit tests for recommendation metrics (AP@K and MAP@K).

Tests exact hand-computed cases, boundary conditions, fewer than K relevant items,
duplicate predictions, and empty prediction/actual inputs.
"""

import pytest

from ranking.metrics import apk, mapk


def test_apk_empty_inputs():
    """Verify empty actual or predicted yields 0.0."""
    assert apk([], ["item_1", "item_2"], k=12) == 0.0
    assert apk(["item_1"], [], k=12) == 0.0
    assert apk([], [], k=12) == 0.0


def test_apk_single_hit_rank_1():
    """1 relevant item predicted at rank 1: P(1)=1.0, denom=1 -> 1.0."""
    actual = ["A"]
    predicted = ["A", "B", "C"]
    assert apk(actual, predicted, k=12) == 1.0


def test_apk_single_hit_rank_2():
    """1 relevant item predicted at rank 2: P(2)=1/2=0.5, denom=1 -> 0.5."""
    actual = ["A"]
    predicted = ["B", "A", "C"]
    assert apk(actual, predicted, k=12) == 0.5


def test_apk_two_hits_rank_1_and_3():
    """2 relevant items at ranks 1 and 3: P(1)=1/1, P(3)=2/3, sum=5/3, denom=2 -> 5/6."""
    actual = ["A", "C"]
    predicted = ["A", "B", "C", "D"]
    expected = (1.0 + 2.0 / 3.0) / 2.0
    assert pytest.approx(apk(actual, predicted, k=12)) == expected


def test_apk_three_hits_even_ranks():
    """3 relevant items at ranks 2, 4, 6: P(2)=1/2, P(4)=2/4=0.5, P(6)=3/6=0.5 -> AP=0.5."""
    actual = ["B", "D", "F"]
    predicted = ["A", "B", "C", "D", "E", "F"]
    expected = (0.5 + 0.5 + 0.5) / 3.0
    assert pytest.approx(apk(actual, predicted, k=12)) == expected


def test_apk_cutoff_at_k():
    """Hits beyond rank k must be ignored."""
    actual = ["X"]
    # 12 irrelevant items, then 'X' at position 13
    predicted = [f"dummy_{i}" for i in range(12)] + ["X"]
    assert apk(actual, predicted, k=12) == 0.0

    # 'X' at position 12 (1-indexed) is included
    predicted_at_12 = [f"dummy_{i}" for i in range(11)] + ["X"]
    expected_at_12 = (1.0 / 12.0) / 1.0
    assert pytest.approx(apk(actual, predicted_at_12, k=12)) == expected_at_12


def test_apk_denominator_capped_at_k():
    """When len(actual) > k (e.g. 16 actual items), denominator is capped at k=12."""
    actual = [f"item_{i}" for i in range(16)]
    # All 12 predictions are hits
    predicted = [f"item_{i}" for i in range(12)]
    # sum(1.0 * 12) / min(16, 12) = 12 / 12 = 1.0
    assert apk(actual, predicted, k=12) == 1.0

    # 6 hits, 6 misses
    predicted_half = [f"item_{i}" for i in range(6)] + [f"wrong_{i}" for i in range(6)]
    # sum = 1/1 + 2/2 + ... + 6/6 = 6.0; denom = 12 -> AP = 6.0 / 12 = 0.5
    assert apk(actual, predicted_half, k=12) == 0.5


def test_apk_duplicate_predictions_handled():
    """Duplicate predictions should not count as multiple hits."""
    actual = ["A"]
    predicted = ["A", "A", "A"]
    # First A is hit at rank 1, remaining A's are duplicate misses
    assert apk(actual, predicted, k=12) == 1.0


def test_mapk_averaging():
    """MAP@K computes exact mean across customers."""
    actuals = [
        ["A"],  # AP = 1.0
        ["A"],  # AP = 0.5
        ["Z"],  # AP = 0.0
    ]
    predictions = [
        ["A", "B"],
        ["B", "A"],
        ["C", "D"],
    ]
    expected_map = (1.0 + 0.5 + 0.0) / 3.0
    assert pytest.approx(mapk(actuals, predictions, k=12)) == expected_map


def test_mapk_filter_empty_actuals():
    """Customers without purchases in target period are filtered if filter_empty_actuals=True."""
    actuals = [
        ["A"],  # AP = 1.0
        [],  # inactive customer
    ]
    predictions = [
        ["A", "B"],
        ["C", "D"],
    ]
    # Filtered: only customer 1 evaluated -> MAP = 1.0
    assert mapk(actuals, predictions, k=12, filter_empty_actuals=True) == 1.0
    # Not filtered: customer 2 counts as 0.0 -> MAP = 0.5
    assert mapk(actuals, predictions, k=12, filter_empty_actuals=False) == 0.5


def test_mapk_length_mismatch_raises():
    """Unequal lengths of actuals and predictions raises ValueError."""
    with pytest.raises(ValueError):
        mapk([["A"]], [["A"], ["B"]])
