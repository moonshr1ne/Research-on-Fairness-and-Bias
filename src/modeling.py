from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier


PROTECTED_COLUMNS = ["Gender", "Age", "AgeGroup"]


def split_xy_generic(
    df: pd.DataFrame,
    target_col: str,
    raw_target_cols: list[str] | None = None,
    drop_protected: bool = False,
    protected_columns: list[str] | None = None,
):
    drop_cols = [target_col]
    if raw_target_cols:
        drop_cols += raw_target_cols
    if drop_protected and protected_columns:
        drop_cols += [c for c in protected_columns if c in df.columns]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target_col].astype(int)
    return X, y


def split_xy(df: pd.DataFrame, drop_protected: bool = False):
    return split_xy_generic(
        df,
        target_col="Attrition_binary",
        raw_target_cols=["Attrition"],
        drop_protected=drop_protected,
        protected_columns=PROTECTED_COLUMNS,
    )


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])


def make_pipeline(model, X: pd.DataFrame) -> Pipeline:
    return Pipeline([
        ("preprocess", make_preprocessor(X)),
        ("model", model),
    ])


def get_models(seed: int = 42) -> dict:
    return {
        "Majority": DummyClassifier(strategy="most_frequent"),
        "LR_all_features": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
        "LR_without_protected": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed,
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=seed),
    }


def predict_from_pipeline(pipe: Pipeline, X: pd.DataFrame, threshold: float = 0.5):
    model = pipe.named_steps["model"]
    if hasattr(model, "predict_proba"):
        y_prob = pipe.predict_proba(X)[:, 1]
    else:
        y_prob = pipe.predict(X).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob


def reweighing_sample_weights(X_train: pd.DataFrame, y_train: pd.Series, sensitive_col: str) -> pd.Series:
    # Kamiran-Calders style reweighing: P(A=a)P(Y=y)/P(A=a,Y=y)
    A = X_train[sensitive_col]
    y = pd.Series(y_train, index=X_train.index)
    weights = pd.Series(1.0, index=X_train.index)
    for a in A.dropna().unique():
        for yy in sorted(y.dropna().unique()):
            idx = (A == a) & (y == yy)
            p_a = (A == a).mean()
            p_y = (y == yy).mean()
            p_ay = idx.mean()
            weights.loc[idx] = (p_a * p_y / p_ay) if p_ay > 0 else 1.0
    return weights


def group_threshold_predictions(y_prob, groups, thresholds_by_group):
    groups = pd.Series(groups).reset_index(drop=True)
    y_prob = np.asarray(y_prob)
    preds = np.zeros(len(groups), dtype=int)
    for g in groups.unique():
        t = thresholds_by_group.get(g, 0.5)
        preds[groups == g] = (y_prob[groups == g] >= t).astype(int)
    return preds


def search_group_thresholds_for_dp(y_prob, groups, base_rate=None, grid=None):
    # Simple post-processing: choose group-specific thresholds to make positive rates close to global target rate.
    groups = pd.Series(groups).reset_index(drop=True)
    y_prob = np.asarray(y_prob)
    if grid is None:
        grid = np.linspace(0.1, 0.9, 81)
    if base_rate is None:
        base_rate = float((y_prob >= 0.5).mean())
    thresholds = {}
    for g in groups.unique():
        best_t, best_err = 0.5, float("inf")
        probs_g = y_prob[groups == g]
        for t in grid:
            rate = float((probs_g >= t).mean())
            err = abs(rate - base_rate)
            if err < best_err:
                best_err, best_t = err, float(t)
        thresholds[g] = best_t
    return thresholds
