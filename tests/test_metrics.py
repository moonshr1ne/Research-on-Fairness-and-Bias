import math

import numpy as np
import pandas as pd

from src.metrics import fairness_metrics, group_rates, max_pairwise_gap, performance_metrics


def test_group_rates_include_confusion_counts():
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 1, 0]
    groups = ["A", "A", "A", "B", "B", "B"]

    rates = group_rates(y_true, y_pred, groups).set_index("group")

    assert rates.loc["A", "n"] == 3
    assert rates.loc["A", "tp"] == 1
    assert rates.loc["A", "fp"] == 1
    assert rates.loc["A", "fn"] == 1
    assert rates.loc["B", "tn"] == 2
    assert rates.loc["B", "positive_prediction_rate"] == 1 / 3


def test_fairness_metrics_use_max_pairwise_gap():
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 1, 0]
    groups = ["A", "A", "A", "B", "B", "B"]

    metrics = fairness_metrics(y_true, y_pred, groups)

    assert math.isclose(metrics["dp_gap"], 1 / 3)
    assert math.isclose(metrics["eo_gap"], 0.5)
    assert math.isclose(metrics["fpr_gap"], 1.0)
    assert math.isclose(metrics["equalized_odds_gap"], 1.0)


def test_max_pairwise_gap_handles_missing_and_single_group():
    assert max_pairwise_gap(pd.Series([0.1, np.nan, 0.4])) == 0.30000000000000004
    assert max_pairwise_gap(pd.Series([0.2])) == 0.0


def test_performance_metrics_handles_constant_probabilities():
    metrics = performance_metrics([0, 1, 1, 0], [0, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])

    assert metrics["accuracy"] == 0.75
    assert metrics["f1"] > 0
    assert np.isnan(metrics["roc_auc"])
