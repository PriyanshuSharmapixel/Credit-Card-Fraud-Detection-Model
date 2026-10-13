# Fraud Detection: Cost-Sensitive ML Upgrade

This branch preserves the original `ccfdm.ipynb` and adds an independent reproducible modeling pipeline.

## Improvements
- Split the original transactions **before** sampling; untouched validation and test prevalence.
- Default chronological 60/20/20 split, or optional stratified split.
- Class-weighted Logistic Regression vs. XGBoost.
- Select model using validation average precision (PR-AUC); tune threshold on validation using assumed false-positive/false-negative costs.
- Report held-out precision, recall, F1, PR-AUC, ROC-AUC, confusion matrix and illustrative cost.
- Export scores and optional XGBoost SHAP feature importance.

## Run
```bash
pip install -r requirements-fraud-upgrade.txt
python fraud_risk_pipeline.py --data /path/to/creditcard.csv --output results
python fraud_risk_pipeline.py --data /path/to/creditcard.csv --split stratified --output results-stratified
```

Requires a local CSV with Time, Amount, V1-V28, and binary Class. Data is not included.

## Limitations
- This is research code, not a production fraud decision engine.
- Class weighting can distort calibration: `fraud_score_uncalibrated` is **not** an absolute fraud probability.
- Costs default to arbitrary example units (FP=10, FN=500); replace with defensible business assumptions.
- SHAP attributes anonymous PCA components, not causal or human-readable fraud reasons.
- Chronological splits may have different fraud rates and are not guaranteed to work on small partial datasets.
- No model performance improvement is claimed until the pipeline is executed on real data.
