# Data

This folder contains the IBM HR Employee Attrition dataset:

`WA_Fn-UseC_-HR-Employee-Attrition.csv`

The pipeline expects exactly this filename. The dataset is used as a public HR analytics case study for reproducible fairness auditing; it should not be interpreted as evidence about a specific employer.

## Optional Employee Promotion Dataset

To add the second HR automation scenario, download Kaggle's HR Analytics Employee Promotion training file and save it as:

`promotion_train.csv`

Expected target column:

`is_promoted`

Expected protected attributes:

`gender`, `age`

The project intentionally does not generate synthetic promotion data. If this file is missing, the main IBM attrition pipeline still runs and the promotion outputs are skipped.
