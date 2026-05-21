# Outputs

Run the full pipeline to generate reproducible CSV artifacts:

```bash
python scripts/run_full_pipeline.py
```

The most important IBM attrition files are `model_metrics.csv`, `group_rates_by_model.csv`, `hypothesis_summary.csv`, and `claims_traceability.csv`.

If `data/promotion_train.csv` is present, the pipeline also generates:

- `promotion_model_metrics.csv`
- `promotion_raw_group_summary.csv`
- `promotion_group_rates_by_model.csv`
- `promotion_summary.csv`
- `cross_dataset_summary.csv`
