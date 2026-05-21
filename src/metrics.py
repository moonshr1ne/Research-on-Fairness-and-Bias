from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def _safe_rate(num: int, den: int) -> float:
    return float(num / den) if den else np.nan


def group_rates(y_true, y_pred, groups) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    groups = pd.Series(groups).reset_index(drop=True)

    rows = []
    for g in sorted(groups.dropna().unique()):
        idx = (groups == g).to_numpy()
        yt = y_true[idx]
        yp = y_pred[idx]
        tp = int(((yt == 1) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        rows.append({
            "group": g,
            "n": int(idx.sum()),
            "positive_prediction_rate": float(yp.mean()) if len(yp) else np.nan,
            "base_rate": float(yt.mean()) if len(yt) else np.nan,
            "tpr": _safe_rate(tp, tp + fn),
            "fpr": _safe_rate(fp, fp + tn),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        })
    return pd.DataFrame(rows)


def max_pairwise_gap(values) -> float:
    values = pd.Series(values).dropna().astype(float)
    if len(values) <= 1:
        return 0.0
    return float(values.max() - values.min())


def fairness_metrics(y_true, y_pred, groups) -> dict:
    rates = group_rates(y_true, y_pred, groups)
    dp_gap = max_pairwise_gap(rates["positive_prediction_rate"])
    eo_gap = max_pairwise_gap(rates["tpr"])
    fpr_gap = max_pairwise_gap(rates["fpr"])
    eodds_gap = max(eo_gap, fpr_gap)
    return {
        "dp_gap": dp_gap,
        "eo_gap": eo_gap,
        "fpr_gap": fpr_gap,
        "equalized_odds_gap": eodds_gap,
        "group_rates": rates,
    }


def performance_metrics(y_true, y_pred, y_prob) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2 and len(np.unique(y_prob)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["roc_auc"] = np.nan
    return out


def full_metrics(y_true, y_pred, y_prob, sensitive_frame: pd.DataFrame) -> dict:
    out = performance_metrics(y_true, y_pred, y_prob)
    for attr in sensitive_frame.columns:
        fm = fairness_metrics(y_true, y_pred, sensitive_frame[attr])
        out[f"{attr}_dp_gap"] = fm["dp_gap"]
        out[f"{attr}_eo_gap"] = fm["eo_gap"]
        out[f"{attr}_fpr_gap"] = fm["fpr_gap"]
        out[f"{attr}_equalized_odds_gap"] = fm["equalized_odds_gap"]
    return out


def bootstrap_ci(metric_func, y_true, y_pred, y_prob, sensitive_frame, n_boot=300, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    records = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sf = sensitive_frame.iloc[idx].reset_index(drop=True)
        rec = metric_func(
            np.asarray(y_true)[idx],
            np.asarray(y_pred)[idx],
            np.asarray(y_prob)[idx],
            sf,
        )
        records.append(rec)
    boot = pd.DataFrame(records)
    rows = []
    for col in boot.columns:
        values = boot[col].dropna()
        if len(values) == 0:
            continue
        rows.append({
            "metric": col,
            "mean": values.mean(),
            "ci_low": values.quantile(0.025),
            "ci_high": values.quantile(0.975),
        })
    return pd.DataFrame(rows)
