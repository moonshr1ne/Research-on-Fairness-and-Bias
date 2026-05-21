from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import OUTPUTS_DIR, PROMOTION_DATA_PATH
from src.data import load_raw_data, add_research_columns, sanity_summary
from src.plotting import make_eda_figures
from src.experiments import (
    make_cross_dataset_summary,
    run_full_experiment,
    run_promotion_full_experiment,
)


def main():
    df = add_research_columns(load_raw_data())
    print("Sanity summary:")
    print(sanity_summary(df))

    print("Generating EDA figures...")
    make_eda_figures(df)

    print("Running baseline, mitigation and fairness experiments...")
    metrics_df, hypothesis_df = run_full_experiment(save=True)

    print("\nModel metrics:")
    print(metrics_df.round(4).to_string(index=False))

    print("\nHypothesis summary:")
    print(hypothesis_df.to_string(index=False))

    print("\nChecking optional Employee Promotion dataset...")
    promotion_result = run_promotion_full_experiment(save=True)
    if promotion_result is None:
        print(f"Promotion dataset not found at {PROMOTION_DATA_PATH}.")
        print("To enable the second HR scenario, save Kaggle HR Analytics Employee Promotion train.csv as data/promotion_train.csv.")
    else:
        promotion_metrics, promotion_summary = promotion_result
        ibm_raw = pd.read_csv(OUTPUTS_DIR / "raw_group_summary.csv")
        promotion_raw = pd.read_csv(OUTPUTS_DIR / "promotion_raw_group_summary.csv")
        cross = make_cross_dataset_summary(metrics_df, ibm_raw, promotion_metrics, promotion_raw)
        cross.to_csv(OUTPUTS_DIR / "cross_dataset_summary.csv", index=False)
        print("\nPromotion model metrics:")
        print(promotion_metrics.round(4).to_string(index=False))
        print("\nPromotion summary:")
        print(promotion_summary.to_string(index=False))
        print("\nCross-dataset summary:")
        print(cross.round(4).to_string(index=False))

    print("\nDone. Outputs saved to outputs/, reports/tables/ and reports/figures/.")


if __name__ == "__main__":
    main()
