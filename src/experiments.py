from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from .config import (
    SEED,
    SENSITIVE_ATTRS,
    OUTPUTS_DIR,
    TABLES_DIR,
    PROMOTION_DATA_PATH,
    PROMOTION_SENSITIVE_ATTRS,
)
from .data import (
    load_raw_data,
    add_research_columns,
    clean_for_modeling,
    load_promotion_data,
    add_promotion_research_columns,
    clean_promotion_for_modeling,
)
from .metrics import full_metrics, bootstrap_ci, group_rates, max_pairwise_gap
from .modeling import (
    get_models,
    make_pipeline,
    predict_from_pipeline,
    split_xy,
    split_xy_generic,
    reweighing_sample_weights,
    search_group_thresholds_for_dp,
    group_threshold_predictions,
)


def prepare_data():
    df = add_research_columns(load_raw_data())
    df_model = clean_for_modeling(df)
    X, y = split_xy(df_model, drop_protected=False)
    stratify_key = y.astype(str) + "_" + X["Gender"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=stratify_key
    )
    return df, X_train, X_test, y_train, y_test


def run_baseline_experiments():
    df, X_train, X_test, y_train, y_test = prepare_data()
    sensitive_test = X_test[SENSITIVE_ATTRS].reset_index(drop=True)

    rows = []
    predictions = {}
    for name, model in get_models(SEED).items():
        drop_protected = name == "LR_without_protected"
        if drop_protected:
            Xtr = X_train.drop(columns=[c for c in ["Gender", "Age", "AgeGroup"] if c in X_train.columns])
            Xte = X_test.drop(columns=[c for c in ["Gender", "Age", "AgeGroup"] if c in X_test.columns])
        else:
            Xtr, Xte = X_train, X_test

        pipe = make_pipeline(model, Xtr)
        pipe.fit(Xtr, y_train)
        y_pred, y_prob = predict_from_pipeline(pipe, Xte)
        metrics = full_metrics(y_test, y_pred, y_prob, sensitive_test)
        metrics["model"] = name
        rows.append(metrics)
        predictions[name] = {"y_pred": y_pred, "y_prob": y_prob}

    return pd.DataFrame(rows), predictions, (df, X_train, X_test, y_train, y_test)


def run_reweighting_experiment():
    df, X_train, X_test, y_train, y_test = prepare_data()
    sensitive_test = X_test[SENSITIVE_ATTRS].reset_index(drop=True)

    # Gender reweighting over A x Y cells.
    weights = reweighing_sample_weights(X_train, y_train, sensitive_col="Gender")
    model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)
    pipe = make_pipeline(model, X_train)
    pipe.fit(X_train, y_train, model__sample_weight=weights)
    y_pred, y_prob = predict_from_pipeline(pipe, X_test)
    metrics = full_metrics(y_test, y_pred, y_prob, sensitive_test)
    metrics["model"] = "LR_reweighting_gender"
    return metrics, {"y_pred": y_pred, "y_prob": y_prob}


def run_postprocessing_experiment():
    df, X_train, X_test, y_train, y_test = prepare_data()
    sensitive_test = X_test[SENSITIVE_ATTRS].reset_index(drop=True)

    model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)
    pipe = make_pipeline(model, X_train)
    pipe.fit(X_train, y_train)
    _, y_prob = predict_from_pipeline(pipe, X_test)
    thresholds = search_group_thresholds_for_dp(y_prob, X_test["Gender"])
    y_pred = group_threshold_predictions(y_prob, X_test["Gender"], thresholds)
    metrics = full_metrics(y_test, y_pred, y_prob, sensitive_test)
    metrics["model"] = "LR_postprocessing_gender_thresholds"
    return metrics, thresholds, {"y_pred": y_pred, "y_prob": y_prob}


def make_raw_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    return make_raw_group_summary_generic(df, SENSITIVE_ATTRS, "Attrition_binary", "attrition_rate")


def make_raw_group_summary_generic(
    df: pd.DataFrame,
    sensitive_attrs: list[str],
    target_col: str,
    rate_col: str,
) -> pd.DataFrame:
    rows = []
    for attr in sensitive_attrs:
        stats = (
            df.groupby(attr)[target_col]
            .agg(n="count", **{rate_col: "mean"})
            .reset_index()
            .rename(columns={attr: "group"})
        )
        gap = max_pairwise_gap(stats[rate_col])
        for rec in stats.to_dict("records"):
            rec["attribute"] = attr
            rec[f"raw_{rate_col}_gap_for_attribute"] = gap
            rows.append(rec)
    return pd.DataFrame(rows)[
        ["attribute", "group", "n", rate_col, f"raw_{rate_col}_gap_for_attribute"]
    ]


def make_model_group_rates(
    predictions: dict,
    y_test: pd.Series,
    sensitive_test: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    y_true = y_test.reset_index(drop=True)
    for model_name, pred in predictions.items():
        for attr in sensitive_test.columns:
            rates = group_rates(y_true, pred["y_pred"], sensitive_test[attr])
            rates.insert(0, "attribute", attr)
            rates.insert(0, "model", model_name)
            rows.append(rates)
    return pd.concat(rows, ignore_index=True)


def make_data_profile(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "missing": [int(df[col].isna().sum()) for col in df.columns],
            "unique_values": [int(df[col].nunique(dropna=True)) for col in df.columns],
            "sample_values": [
                ", ".join(map(str, df[col].dropna().unique()[:5])) for col in df.columns
            ],
        }
    )


def make_report_claims(
    df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    hypothesis_df: pd.DataFrame,
    raw_group_summary: pd.DataFrame,
) -> pd.DataFrame:
    metrics = metrics_df.set_index("model")
    rows = [
        {
            "claim_id": "DATA-001",
            "claim": "The IBM HR Employee Attrition dataset contains employee records used for attrition prediction.",
            "value": f"{len(df)} rows, {df.shape[1]} columns",
            "source_file": "outputs/data_profile.csv",
            "source_filter": "all rows",
        },
        {
            "claim_id": "EDA-001",
            "claim": "The target is imbalanced, so accuracy alone is insufficient.",
            "value": f"Attrition=Yes rate: {df['Attrition_binary'].mean():.3f}",
            "source_file": "outputs/raw_group_summary.csv",
            "source_filter": "overall target distribution from source data",
        },
    ]

    for attr in SENSITIVE_ATTRS:
        sub = raw_group_summary[raw_group_summary["attribute"] == attr]
        rates = ", ".join(
            f"{r.group}: {r.attrition_rate:.3f}" for r in sub.itertuples(index=False)
        )
        gap = float(sub["raw_attrition_rate_gap_for_attribute"].iloc[0])
        rows.append(
            {
                "claim_id": f"EDA-{attr.upper()}",
                "claim": f"Raw attrition rates differ across {attr} groups.",
                "value": f"{rates}; max gap={gap:.3f}",
                "source_file": "outputs/raw_group_summary.csv",
                "source_filter": f"attribute == {attr}",
            }
        )

    if "LR_all_features" in metrics.index:
        baseline = metrics.loc["LR_all_features"]
        rows.append(
            {
                "claim_id": "MODEL-BASELINE",
                "claim": "The main baseline is Logistic Regression with protected attributes included.",
                "value": (
                    f"ROC-AUC={baseline['roc_auc']:.3f}, F1={baseline['f1']:.3f}, "
                    f"Gender DP gap={baseline['Gender_dp_gap']:.3f}, "
                    f"AgeGroup DP gap={baseline['AgeGroup_dp_gap']:.3f}"
                ),
                "source_file": "outputs/model_metrics.csv",
                "source_filter": "model == LR_all_features",
            }
        )

    for row in hypothesis_df.itertuples(index=False):
        rows.append(
            {
                "claim_id": row.hypothesis.split()[0],
                "claim": row.pre_registered_rule,
                "value": f"observed={row.observed_value}; decision={row.decision}",
                "source_file": "outputs/hypothesis_summary.csv",
                "source_filter": f"hypothesis == {row.hypothesis}",
            }
        )

    return pd.DataFrame(rows)


def write_claims_markdown(claims_df: pd.DataFrame, path):
    lines = [
        "# Claims traceability map",
        "",
        "Every statement below is generated from the source dataset and model outputs.",
        "Use this file to connect report text to reproducible CSV artifacts.",
        "",
        "| Claim ID | Claim | Value | Source |",
        "|---|---|---|---|",
    ]
    for row in claims_df.itertuples(index=False):
        source = f"{row.source_file} ({row.source_filter})"
        lines.append(f"| {row.claim_id} | {row.claim} | {row.value} | {source} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_promotion_data():
    df = add_promotion_research_columns(load_promotion_data())
    df_model = clean_promotion_for_modeling(df)
    X, y = split_xy_generic(
        df_model,
        target_col="Promotion_binary",
        raw_target_cols=["is_promoted"],
        drop_protected=False,
    )
    stratify_key = y.astype(str) + "_" + X["gender"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=stratify_key
    )
    return df, X_train, X_test, y_train, y_test


def run_promotion_baseline_experiments():
    df, X_train, X_test, y_train, y_test = prepare_promotion_data()
    sensitive_test = X_test[PROMOTION_SENSITIVE_ATTRS].reset_index(drop=True)

    rows = []
    predictions = {}
    protected = ["gender", "age", "AgeGroup"]
    for name, model in get_models(SEED).items():
        drop_protected = name == "LR_without_protected"
        if drop_protected:
            Xtr = X_train.drop(columns=[c for c in protected if c in X_train.columns])
            Xte = X_test.drop(columns=[c for c in protected if c in X_test.columns])
        else:
            Xtr, Xte = X_train, X_test

        pipe = make_pipeline(model, Xtr)
        pipe.fit(Xtr, y_train)
        y_pred, y_prob = predict_from_pipeline(pipe, Xte)
        metrics = full_metrics(y_test, y_pred, y_prob, sensitive_test)
        metrics["model"] = name
        rows.append(metrics)
        predictions[name] = {"y_pred": y_pred, "y_prob": y_prob}

    return pd.DataFrame(rows), predictions, (df, X_train, X_test, y_train, y_test)


def run_promotion_reweighting_experiment():
    df, X_train, X_test, y_train, y_test = prepare_promotion_data()
    sensitive_test = X_test[PROMOTION_SENSITIVE_ATTRS].reset_index(drop=True)

    weights = reweighing_sample_weights(X_train, y_train, sensitive_col="gender")
    model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)
    pipe = make_pipeline(model, X_train)
    pipe.fit(X_train, y_train, model__sample_weight=weights)
    y_pred, y_prob = predict_from_pipeline(pipe, X_test)
    metrics = full_metrics(y_test, y_pred, y_prob, sensitive_test)
    metrics["model"] = "LR_reweighting_gender"
    return metrics, {"y_pred": y_pred, "y_prob": y_prob}


def make_promotion_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    m = metrics_df.set_index("model")
    rows = []
    if "LR_all_features" in m.index:
        gender_gap = float(m.loc["LR_all_features", "gender_dp_gap"])
        age_gap = float(m.loc["LR_all_features", "AgeGroup_dp_gap"])
        rows.append({
            "check": "Promotion baseline fairness",
            "observed_value": f"gender_dp_gap={gender_gap:.3f}, agegroup_dp_gap={age_gap:.3f}",
            "interpretation": "AgeGroup gap is larger" if age_gap > gender_gap else "Gender gap is larger or equal",
        })
    if "LR_all_features" in m.index and "LR_without_protected" in m.index:
        before = float(m.loc["LR_all_features", "gender_dp_gap"])
        after = float(m.loc["LR_without_protected", "gender_dp_gap"])
        reduction = (before - after) / before if before > 0 else np.nan
        rows.append({
            "check": "Promotion proxy persistence",
            "observed_value": f"gender_dp_reduction={reduction:.3f}",
            "interpretation": "Protected-feature removal reduced the gap" if reduction > 0 else "Gap persisted or increased after protected-feature removal",
        })
    return pd.DataFrame(rows)


def make_cross_dataset_summary(
    ibm_metrics: pd.DataFrame,
    ibm_raw: pd.DataFrame,
    promotion_metrics: pd.DataFrame | None = None,
    promotion_raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    def weighted_rate(raw: pd.DataFrame, attribute: str, rate_col: str) -> float:
        sub = raw[raw["attribute"] == attribute]
        return float((sub["n"] * sub[rate_col]).sum() / sub["n"].sum())

    rows = []
    ibm_base = ibm_metrics.set_index("model").loc["LR_all_features"]
    rows.append({
        "dataset": "IBM HR Attrition",
        "task": "predict employee attrition",
        "target_positive_rate": weighted_rate(ibm_raw, "Gender", "attrition_rate"),
        "baseline_roc_auc": float(ibm_base["roc_auc"]),
        "baseline_f1": float(ibm_base["f1"]),
        "gender_dp_gap": float(ibm_base["Gender_dp_gap"]),
        "agegroup_dp_gap": float(ibm_base["AgeGroup_dp_gap"]),
    })

    if promotion_metrics is not None and promotion_raw is not None:
        promotion_base = promotion_metrics.set_index("model").loc["LR_all_features"]
        rows.append({
            "dataset": "Employee Promotion",
            "task": "predict promotion recommendation",
            "target_positive_rate": weighted_rate(promotion_raw, "gender", "promotion_rate"),
            "baseline_roc_auc": float(promotion_base["roc_auc"]),
            "baseline_f1": float(promotion_base["f1"]),
            "gender_dp_gap": float(promotion_base["gender_dp_gap"]),
            "agegroup_dp_gap": float(promotion_base["AgeGroup_dp_gap"]),
        })

    return pd.DataFrame(rows)


def run_promotion_full_experiment(save=True):
    if not PROMOTION_DATA_PATH.exists():
        return None

    baseline_df, predictions, data_pack = run_promotion_baseline_experiments()
    rw_metrics, rw_pred = run_promotion_reweighting_experiment()
    metrics_df = pd.concat([baseline_df, pd.DataFrame([rw_metrics])], ignore_index=True)
    predictions["LR_reweighting_gender"] = rw_pred
    promotion_summary = make_promotion_summary(metrics_df)

    if save:
        df, X_train, X_test, y_train, y_test = data_pack
        raw_group_summary = make_raw_group_summary_generic(
            df, PROMOTION_SENSITIVE_ATTRS, "Promotion_binary", "promotion_rate"
        )
        model_group_rates = make_model_group_rates(
            predictions,
            y_test.reset_index(drop=True),
            X_test[PROMOTION_SENSITIVE_ATTRS].reset_index(drop=True),
        )

        metrics_df.to_csv(OUTPUTS_DIR / "promotion_model_metrics.csv", index=False)
        raw_group_summary.to_csv(OUTPUTS_DIR / "promotion_raw_group_summary.csv", index=False)
        model_group_rates.to_csv(OUTPUTS_DIR / "promotion_group_rates_by_model.csv", index=False)
        promotion_summary.to_csv(OUTPUTS_DIR / "promotion_summary.csv", index=False)
        metrics_df.round(4).to_csv(TABLES_DIR / "promotion_model_metrics_table.csv", index=False)
        raw_group_summary.round(4).to_csv(TABLES_DIR / "promotion_raw_group_summary_table.csv", index=False)
        model_group_rates.round(4).to_csv(TABLES_DIR / "promotion_group_rates_by_model_table.csv", index=False)
        promotion_summary.to_csv(TABLES_DIR / "promotion_summary_table.csv", index=False)

    return metrics_df, promotion_summary


def make_hypothesis_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    m = metrics_df.set_index("model")
    rows = []

    if "LR_all_features" in m.index and "LR_without_protected" in m.index:
        before = float(m.loc["LR_all_features", "Gender_dp_gap"])
        after = float(m.loc["LR_without_protected", "Gender_dp_gap"])
        reduction = (before - after) / before if before > 0 else np.nan
        rows.append({
            "hypothesis": "H1 Protected-feature removal test",
            "pre_registered_rule": "Removing protected features reduces Gender DP gap by at least 80%.",
            "observed_value": f"before={before:.3f}, after={after:.3f}, reduction={reduction:.3f}",
            "decision": "supported" if pd.notna(reduction) and reduction >= 0.80 else "falsified",
        })

    if "LR_all_features" in m.index:
        gender_gap = float(m.loc["LR_all_features", "Gender_dp_gap"])
        age_gap = float(m.loc["LR_all_features", "AgeGroup_dp_gap"])
        rows.append({
            "hypothesis": "H2 Age disparity dominance",
            "pre_registered_rule": "AgeGroup DP gap exceeds Gender DP gap by at least 0.05.",
            "observed_value": age_gap - gender_gap,
            "decision": "supported" if (age_gap - gender_gap) >= 0.05 else "not supported",
        })

    if "LR_all_features" in m.index and "LR_reweighting_gender" in m.index:
        before_gap = float(m.loc["LR_all_features", "Gender_dp_gap"])
        after_gap = float(m.loc["LR_reweighting_gender", "Gender_dp_gap"])
        gap_reduction = (before_gap - after_gap) / before_gap if before_gap > 0 else np.nan
        auc_drop = float(m.loc["LR_all_features", "roc_auc"] - m.loc["LR_reweighting_gender", "roc_auc"])
        rows.append({
            "hypothesis": "H3 Mitigation trade-off",
            "pre_registered_rule": "Reweighting reduces Gender DP gap by at least 25% and decreases ROC-AUC or F1.",
            "observed_value": f"gap_reduction={gap_reduction:.3f}, auc_drop={auc_drop:.3f}",
            "decision": "supported" if pd.notna(gap_reduction) and gap_reduction >= 0.25 and auc_drop > 0 else "not supported / partially falsified",
        })

    if "LR_all_features" in m.index and "RandomForest" in m.index:
        auc_gain = float(m.loc["RandomForest", "roc_auc"] - m.loc["LR_all_features", "roc_auc"])
        fairness_change = float(m.loc["RandomForest", "Gender_dp_gap"] - m.loc["LR_all_features", "Gender_dp_gap"])
        rows.append({
            "hypothesis": "H4 Model complexity trade-off",
            "pre_registered_rule": "RandomForest improves ROC-AUC by at least 0.02 but does not necessarily improve fairness.",
            "observed_value": f"auc_gain={auc_gain:.3f}, gender_dp_change={fairness_change:.3f}",
            "decision": "supported" if auc_gain >= 0.02 else "not supported",
        })

    return pd.DataFrame(rows)


def run_full_experiment(save=True):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    baseline_df, predictions, data_pack = run_baseline_experiments()
    rw_metrics, rw_pred = run_reweighting_experiment()
    pp_metrics, thresholds, pp_pred = run_postprocessing_experiment()

    metrics_df = pd.concat([baseline_df, pd.DataFrame([rw_metrics]), pd.DataFrame([pp_metrics])], ignore_index=True)
    hypothesis_df = make_hypothesis_summary(metrics_df)
    predictions["LR_reweighting_gender"] = rw_pred
    predictions["LR_postprocessing_gender_thresholds"] = pp_pred

    if save:
        metrics_df.to_csv(OUTPUTS_DIR / "model_metrics.csv", index=False)
        hypothesis_df.to_csv(OUTPUTS_DIR / "hypothesis_summary.csv", index=False)
        metrics_df.round(4).to_csv(TABLES_DIR / "model_metrics_table.csv", index=False)
        hypothesis_df.to_csv(TABLES_DIR / "hypothesis_summary_table.csv", index=False)

        df, X_train, X_test, y_train, y_test = data_pack
        raw_group_summary = make_raw_group_summary(df)
        model_group_rates = make_model_group_rates(
            predictions,
            y_test.reset_index(drop=True),
            X_test[SENSITIVE_ATTRS].reset_index(drop=True),
        )
        claims_df = make_report_claims(df, metrics_df, hypothesis_df, raw_group_summary)

        make_data_profile(df).to_csv(OUTPUTS_DIR / "data_profile.csv", index=False)
        raw_group_summary.to_csv(OUTPUTS_DIR / "raw_group_summary.csv", index=False)
        model_group_rates.to_csv(OUTPUTS_DIR / "group_rates_by_model.csv", index=False)
        claims_df.to_csv(OUTPUTS_DIR / "claims_traceability.csv", index=False)
        write_claims_markdown(claims_df, OUTPUTS_DIR / "claims_for_report.md")
        pd.DataFrame(
            [{"attribute": "Gender", "group": group, "threshold": value} for group, value in thresholds.items()]
        ).to_csv(OUTPUTS_DIR / "postprocessing_thresholds.csv", index=False)

        raw_group_summary.round(4).to_csv(TABLES_DIR / "raw_group_summary_table.csv", index=False)
        model_group_rates.round(4).to_csv(TABLES_DIR / "group_rates_by_model_table.csv", index=False)
        claims_df.to_csv(TABLES_DIR / "claims_traceability_table.csv", index=False)
        make_cross_dataset_summary(metrics_df, raw_group_summary).to_csv(
            OUTPUTS_DIR / "cross_dataset_summary.csv", index=False
        )

        # Bootstrap CI for LR_all_features keeps the uncertainty estimate compact and auditable.
        sf = X_test[SENSITIVE_ATTRS].reset_index(drop=True)
        y_pred = predictions["LR_all_features"]["y_pred"]
        y_prob = predictions["LR_all_features"]["y_prob"]
        ci = bootstrap_ci(full_metrics, y_test.reset_index(drop=True), y_pred, y_prob, sf, n_boot=300, seed=SEED)
        ci["model"] = "LR_all_features"
        ci.to_csv(OUTPUTS_DIR / "bootstrap_ci.csv", index=False)

    return metrics_df, hypothesis_df


