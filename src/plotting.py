from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import FIGURES_DIR


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def proportion_ci(p: pd.Series, n: pd.Series, z: float = 1.96) -> pd.Series:
    return z * np.sqrt((p * (1 - p)) / n)


def make_eda_figures(df: pd.DataFrame, figures_dir: Path = FIGURES_DIR):
    figures_dir.mkdir(parents=True, exist_ok=True)

    attr_counts = df["Attrition"].value_counts().reindex(["No", "Yes"])
    plt.figure(figsize=(7, 4))
    plt.bar(attr_counts.index, attr_counts.values)
    plt.title("Overall Attrition Distribution")
    plt.xlabel("Attrition")
    plt.ylabel("Count")
    savefig(figures_dir / "fig_attrition_distribution.png")

    gender_prop = pd.crosstab(df["Gender"], df["Attrition"], normalize="index").reindex(["Female", "Male"])
    plt.figure(figsize=(7, 4))
    plt.bar(gender_prop.index, gender_prop["No"], label="No")
    plt.bar(gender_prop.index, gender_prop["Yes"], bottom=gender_prop["No"], label="Yes")
    plt.title("Proportion of Attrition by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Proportion")
    plt.ylim(0, 1)
    plt.legend()
    savefig(figures_dir / "fig_attrition_by_gender_stacked.png")

    gender_rate = df.groupby("Gender")["Attrition_binary"].mean().reindex(["Female", "Male"])
    gender_n = df.groupby("Gender")["Attrition_binary"].count().reindex(["Female", "Male"])
    gender_ci = proportion_ci(gender_rate, gender_n)
    plt.figure(figsize=(7, 4))
    plt.bar(gender_rate.index, gender_rate.values, yerr=gender_ci.values, capsize=6)
    plt.title("Attrition Rate by Gender (95% CI)")
    plt.xlabel("Gender")
    plt.ylabel("Attrition Rate")
    savefig(figures_dir / "fig_attrition_by_gender_ci.png")

    age_order = ["<30", "30-40", "40+"]
    age_rate = df.groupby("AgeGroup")["Attrition_binary"].mean().reindex(age_order)
    age_n = df.groupby("AgeGroup")["Attrition_binary"].count().reindex(age_order)
    age_ci = proportion_ci(age_rate, age_n)
    plt.figure(figsize=(7, 4))
    plt.bar(age_rate.index.astype(str), age_rate.values, yerr=age_ci.values, capsize=6)
    plt.title("Attrition Rate by Age Group (95% CI)")
    plt.xlabel("Age Group")
    plt.ylabel("Attrition Rate")
    savefig(figures_dir / "fig_attrition_by_agegroup_ci.png")

    plt.figure(figsize=(8, 4))
    for g in ["Female", "Male"]:
        sub = df[df["Gender"] == g]
        plt.hist(sub["MonthlyIncome"], bins=30, alpha=0.5, label=g)
    plt.title("Monthly Income Distribution by Gender")
    plt.xlabel("Monthly Income")
    plt.ylabel("Frequency")
    plt.legend()
    savefig(figures_dir / "fig_income_distribution_by_gender.png")

    ot_prop = pd.crosstab([df["OverTime"], df["Gender"]], df["Attrition"], normalize="index")
    index_order = []
    for ot in ["No", "Yes"]:
        for g in ["Female", "Male"]:
            if (ot, g) in ot_prop.index:
                index_order.append((ot, g))
    ot_prop = ot_prop.reindex(index_order)
    labels = [f"OverTime={ot}, {g}" for ot, g in ot_prop.index]
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(labels)), ot_prop["No"], label="No")
    plt.bar(range(len(labels)), ot_prop["Yes"], bottom=ot_prop["No"], label="Yes")
    plt.title("Proportion of Attrition by OverTime and Gender")
    plt.ylabel("Proportion")
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.legend()
    savefig(figures_dir / "fig_attrition_overtime_gender.png")

    num = df.select_dtypes(include=["int64", "float64"]).copy()
    corr = num.corr()
    plt.figure(figsize=(11, 9))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Matrix (Numeric Features)")
    savefig(figures_dir / "fig_correlation_matrix_numeric.png")
