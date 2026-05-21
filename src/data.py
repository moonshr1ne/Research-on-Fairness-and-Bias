from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import DATA_PATH, DROP_COLS, PROMOTION_DATA_PATH, PROMOTION_DROP_COLS


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Put WA_Fn-UseC_-HR-Employee-Attrition.csv into the data/ folder."
        )
    return pd.read_csv(path)


def load_promotion_data(path: Path = PROMOTION_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Promotion dataset not found: {path}. Download the HR Analytics Employee Promotion train.csv "
            "and save it as data/promotion_train.csv."
        )
    return pd.read_csv(path)


def add_research_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Attrition_binary"] = (df["Attrition"] == "Yes").astype(int)
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 29, 39, 120],
        labels=["<30", "30-40", "40+"],
        include_lowest=True,
    ).astype(str)
    return df


def add_promotion_research_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = {"is_promoted", "age", "gender"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Promotion dataset is missing required columns: {sorted(missing)}")

    df["Promotion_binary"] = df["is_promoted"].astype(int)
    df["AgeGroup"] = pd.cut(
        df["age"],
        bins=[0, 29, 39, 120],
        labels=["<30", "30-40", "40+"],
        include_lowest=True,
    ).astype(str)
    return df


def clean_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def clean_promotion_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_to_drop = [c for c in PROMOTION_DROP_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def sanity_summary(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicated_rows": int(df.duplicated().sum()),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
