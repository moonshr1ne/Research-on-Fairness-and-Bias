# Fairness and Bias in ML Models for HR Automation

This repository supports the course project **Research on Fairness and Bias in Machine Learning Models for HR Automation**. Its purpose is practical: every numerical claim in the report should be traceable to executable code and generated artifacts.

The project uses the IBM HR Employee Attrition dataset to study whether attrition prediction models show measurable demographic disparities across `Gender` and `AgeGroup`, and whether mitigation methods reduce those disparities without unacceptable predictive-performance loss. It also supports an optional second HR automation scenario, Employee Promotion Prediction, for checking whether conclusions generalize beyond attrition.

## Research Question

Do HR-related machine learning models show measurable demographic disparities, and can bias mitigation methods reduce these disparities without unacceptable loss of predictive performance?

## Pre-Registered Hypotheses

- **H1. Proxy persistence:** removing protected attributes (`Gender`, `Age`, `AgeGroup`) will not eliminate demographic parity gaps completely; the Gender DP gap should decrease by no more than 30% relative to the model with protected attributes.
- **H2. Age disparity dominance:** the AgeGroup demographic parity gap will exceed the Gender demographic parity gap by at least 5 percentage points for the baseline model.
- **H3. Mitigation trade-off:** reweighting will reduce the Gender demographic parity gap by at least 25%, but will reduce ROC-AUC or F1 compared with the unmitigated baseline.
- **H4. Model complexity trade-off:** a tree-based model will improve ROC-AUC by at least 0.02 compared with Logistic Regression, but will not necessarily improve fairness gaps.

## Repository Structure

```text
fairness_hr_automation_repo/
|-- README.md
|-- requirements.txt
|-- data/
|   |-- WA_Fn-UseC_-HR-Employee-Attrition.csv
|   |-- promotion_train.csv              # optional: Kaggle Employee Promotion train.csv
|   `-- README.md
|-- notebooks/
|   `-- 01_fairness_hr_research.ipynb
|-- src/
|   |-- config.py
|   |-- data.py
|   |-- metrics.py
|   |-- modeling.py
|   |-- plotting.py
|   `-- experiments.py
|-- scripts/
|   `-- run_full_pipeline.py
|-- tests/
|   `-- test_metrics.py
|-- reports/
|   |-- main.tex
|   |-- figures/
|   `-- tables/
`-- outputs/
```

## How to Reproduce

1. Open this folder in VS Code or a terminal.
2. Create and activate a virtual environment.

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Optional: to enable the second dataset, download the Kaggle Employee Promotion training file and save it as:

```text
data/promotion_train.csv
```

The expected columns include `employee_id`, `department`, `region`, `education`, `gender`, `age`, `avg_training_score`, and `is_promoted`.

5. Run the full pipeline.

```bash
python scripts/run_full_pipeline.py
```

6. Optional: run the tests.

```bash
pytest
```

## Main Generated Artifacts

- `outputs/data_profile.csv` - dataset columns, types, missing values and sample values.
- `outputs/raw_group_summary.csv` - raw attrition rates by `Gender` and `AgeGroup`.
- `outputs/model_metrics.csv` - accuracy, F1, ROC-AUC and fairness gaps for every model.
- `outputs/group_rates_by_model.csv` - group-level base rates, prediction rates, TPR/FPR and confusion counts.
- `outputs/bootstrap_ci.csv` - bootstrap confidence intervals for the main Logistic Regression baseline.
- `outputs/hypothesis_summary.csv` - hypothesis-level decisions with observed values.
- `outputs/claims_traceability.csv` and `outputs/claims_for_report.md` - report claims mapped to the exact output files that support them.
- `outputs/promotion_model_metrics.csv` - generated only when `data/promotion_train.csv` is present.
- `outputs/promotion_group_rates_by_model.csv` - group-level fairness details for the promotion dataset.
- `outputs/cross_dataset_summary.csv` - comparison of attrition and promotion scenarios.
- `reports/figures/` - EDA figures used in the report.
- `reports/tables/` - rounded CSV tables ready to cite or convert for the report.

## Optional Second Dataset

The supported second dataset is Kaggle's **HR Analytics: Employee Promotion Data / HR Analysis Case Study**. It is useful because promotion recommendation is a high-stakes HR decision and contains both `gender` and `age`, allowing the same fairness audit logic to be reused for a different HR task.

The pipeline does not create synthetic promotion data. If `data/promotion_train.csv` is absent, the IBM attrition analysis still runs and the script prints instructions for enabling the second scenario.

## Notes for the Report

The repository does not claim that the IBM dataset proves real-world discrimination by a specific employer. It is used as a controlled HR analytics case study. The correct interpretation is: if group-level outcome rates, prediction rates or error rates differ, the model requires fairness analysis before deployment in a high-stakes HR setting.
