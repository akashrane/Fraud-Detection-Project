# Credit Card Default Risk Modeling

Predict whether a customer will default on a credit card using the [AmExpert 2021 dataset mirrored on Kaggle](https://www.kaggle.com/datasets/hotsonhonet/amex-competition). The project explores customer attributes, engineers a few financial features, and compares three classifiers with an evaluation pipeline designed to avoid train/test leakage.

> **Scope:** The label is `credit_card_default` for each customer. This is **customer default prediction**, not transaction fraud detection or the separate American Express Default Prediction competition.

## Results

The modeling section ran on all **45,528 labeled customers** in `dataset/train.csv`. A stratified 60/20/20 train/validation/test split used `random_state=42`. Preprocessing and SMOTE were fitted only on training data. The model was selected by **validation PR-AUC**, then evaluated once on the untouched **9,106-customer test set** at a fixed 0.5 threshold.

| Model | Validation PR-AUC | Validation ROC-AUC | Validation F1 |
| --- | ---: | ---: | ---: |
| XGBoost | 0.9483 | 0.9943 | 0.8007 |
| Random Forest | 0.9474 | 0.9943 | 0.8415 |
| Logistic Regression | 0.9390 | 0.9915 | 0.7665 |

| Untouched test metric, selected XGBoost | Value |
| --- | ---: |
| Accuracy | 0.9616 |
| Precision | 0.6912 |
| Recall | 0.9513 |
| F1 | 0.8007 |
| ROC-AUC | 0.9951 |
| PR-AUC (average precision) | 0.9559 |
| False-positive rate | 0.0375 |
| True negatives / false positives / false negatives / true positives | 8053 / 314 / 36 / 703 |

The test set's positive-class prevalence was **8.12%**. These numbers describe this dataset and split; they are not a Kaggle leaderboard score or evidence of performance on new populations.

## Method

1. Load the labeled CSV and create row-wise features such as `in_hand_balance` and `employment_years`. Exclude `customer_id` and `name`.
2. Split customers into stratified training, validation, and test partitions before any fitted preprocessing.
3. Fit median/mode imputation, numeric scaling, categorical one-hot encoding, and SMOTE **inside each training pipeline**. Validation and test rows retain their original class distribution.
4. Compare Logistic Regression, Random Forest, and XGBoost on validation PR-AUC. XGBoost is optional when the package is unavailable.
5. Evaluate the selected pipeline on the untouched test set using precision, recall, F1, ROC-AUC, PR-AUC, false-positive rate, and a confusion matrix.
6. Inspect validation-set permutation importance. It describes global associations with PR-AUC, **not causal or individual customer explanations**.

## Run the notebook

1. Download the [dataset](https://www.kaggle.com/datasets/hotsonhonet/amex-competition) and locate `dataset/train.csv`. On Kaggle, attach it as an input; the notebook expects `/kaggle/input/amex-competition/dataset/train.csv`. Locally, change `DATA_PATH` in the modeling section (and the earlier EDA load path).
2. Install dependencies:

   ```bash
   python -m pip install pandas numpy matplotlib scikit-learn imbalanced-learn xgboost jupyter
   ```

3. Open [Fraud Detection With Amex Data.ipynb](Fraud%20Detection%20With%20Amex%20Data.ipynb) and run the cells in order. The earlier EDA charts are exploratory; the modeling section reloads the raw labeled CSV so its preprocessing is learned only from training data.

The Kaggle data is not committed to this repository. Re-run the notebook in your environment to reproduce the metrics; package versions can affect small numerical differences.
