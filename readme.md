# Credit Card Fraud Detection

A reproducible starting point for classifying fraudulent card transactions with logistic regression. The notebook examines class imbalance, samples a smaller training dataset, scales the features, and evaluates a binary classifier. It is a portfolio experiment, not a deployed fraud screening system.

## At a glance

| Item | What the repository shows |
| --- | --- |
| Task | Predict `Class`: `0` for legitimate and `1` for fraud |
| Data | [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (linked in `Dataset_Link.txt`) |
| Model | `StandardScaler` followed by `LogisticRegression(max_iter=1000)` |
| Sample used for modeling | 492 legitimate and 198 fraudulent transactions (690 total) |
| Evaluation split | Stratified 80/20 split; 552 training and 138 test rows; `random_state=2` for the split |
| Reported result | 97.46% training accuracy and 92.75% test accuracy on the sampled dataset |

## Problem and data

Fraud detection has a strong class imbalance: the notebook's loaded CSV contains 81,100 legitimate and 198 fraudulent labeled rows. A classifier that predicts every transaction as legitimate would achieve about 99.76% accuracy on those labeled rows while detecting no fraud. This is why accuracy alone is not an adequate measure of a fraud detector.

The source dataset provides `Time`, `Amount`, anonymized features `V1`–`V28`, and the target `Class`. The notebook output shows **81,299 loaded rows**, with the last row incomplete and missing its target. It is therefore a partial or truncated copy of the linked dataset, not evidence that the full Kaggle dataset was analyzed. The incomplete row is excluded from the modeling sample because it has no class label; the notebook does not explicitly clean or validate the entire CSV.

## What the notebook does

1. Loads `/content/creditcard.csv` and inspects the first and last rows, data types, missing values, class counts, and amount summaries.
2. Separates legitimate and fraudulent transactions.
3. Samples **492** legitimate rows and combines them with **198** fraud rows. This yields 690 rows with a much higher fraud prevalence than in the loaded data.
4. Uses all 30 input columns (`Time`, `V1`–`V28`, and `Amount`) to predict `Class`.
5. Creates a stratified 80/20 train/test split with `random_state=2`.
6. Fits `StandardScaler` on the training features, transforms both splits, and trains logistic regression with `max_iter=1000`.
7. Reports training and test accuracy at the classifier's default decision threshold.

The code and saved outputs are in [`ccfdm.ipynb`](ccfdm.ipynb). The repository currently contains one model; it does not run automated model comparison or model selection.

## Results and interpretation

| Metric | Saved notebook output | Scope |
| --- | ---: | --- |
| Training accuracy | 97.46% | 552 sampled rows |
| Test accuracy | 92.75% | 138 sampled rows |

The test result corresponds to **128 correct predictions out of 138**, but the notebook does not report how many frauds were caught or how many legitimate transactions were flagged. Precision, recall, F1, PR-AUC, ROC-AUC, and a confusion matrix are not calculated. These accuracy values must **not** be presented as performance on the full, naturally imbalanced dataset or as evidence of production readiness.

Sampling happens **before** the train/test split. That means the test set inherits an artificial fraud prevalence of roughly 29%, and performance under real transaction prevalence remains unknown. Also, `legit.sample(n=492)` has no fixed random seed, so rerunning the notebook may change the sample and its reported accuracy even though the split has a fixed seed.

## Run the notebook

1. Clone the repository and install the libraries used by the notebook:

   ```bash
   git clone https://github.com/PriyanshuSharmapixel/Credit-Card-Fraud-Detection-Model.git
   cd Credit-Card-Fraud-Detection-Model
   python -m pip install numpy pandas scikit-learn jupyter
   ```

2. Download `creditcard.csv` from the [dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The dataset file is not committed to this repository.
3. Open `ccfdm.ipynb` in Google Colab or Jupyter. For the notebook **as written**, place the CSV at `/content/creditcard.csv` (for example, upload it to Colab). To run locally, change the `pd.read_csv('/content/creditcard.csv')` line to your local CSV path.
4. Run the cells in order. A full dataset download will not reproduce the saved 81,299-row input or necessarily the saved metrics; those outputs came from the partial CSV and an unseeded sample.

There is no `requirements.txt`, command-line training script, saved model, API, or deployment configuration in this repository.

## Evaluation work needed for an industry-ready version

- Validate the complete dataset and document the exact input version and row counts. Reject malformed rows explicitly.
- Hold out a test set with the **original fraud prevalence before any undersampling**. Sample or rebalance training data only, with a fixed random seed.
- Report a confusion matrix, fraud-class precision and recall, F1, PR-AUC, and alert volume at a chosen threshold. Compare against a clear baseline.
- Choose the threshold on validation data according to the cost of missed fraud versus false alerts; evaluate once on the untouched test set.
- Test a time-based holdout if transaction chronology is available, and check probability calibration and performance drift before considering operational use.

## Repository contents

| File | Purpose |
| --- | --- |
| [`ccfdm.ipynb`](ccfdm.ipynb) | Data inspection, sampled training, scaling, logistic regression, and accuracy outputs |
| [`Dataset_Link.txt`](Dataset_Link.txt) | Link to the source dataset |
| `README.md` | Project overview and reproduction guidance |

## Scope

This notebook demonstrates an initial fraud classification workflow and highlights why class imbalance affects evaluation. It does not establish a reliable fraud detection system for financial decisions. In particular, its saved test accuracy is measured on a small, artificially sampled holdout.
