#  Credit Card Fraud Detection using Machine Learning

This project tackles the challenge of detecting fraudulent credit card transactions in a highly imbalanced dataset using advanced machine learning techniques. The full process includes data preprocessing, class balancing, model training, evaluation, and pipeline development — all implemented in Python and google Colab.

---

# Project Overview

Credit card fraud is a growing global issue, costing financial institutions billions annually. Detecting such rare events among thousands of legitimate transactions requires highly sensitive models that prioritize recall and precision over accuracy.

This project uses LightGBM, a high-performance gradient boosting algorithm, and SMOTE, a popular oversampling technique, to address the class imbalance problem and improve model performance on fraud detection tasks.

---

## Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraud cases:** 492 (~0.17%)
- **Features:**
  - `Time`, `Amount`, `Class` (target)
  - `V1`–`V28`: PCA-transformed anonymized features


## Methodology

- Removed duplicate records to ensure data quality
- Applied `StandardScaler` to normalize numerical features
- Split data into:
  - 60% for development (model training & validation)
  - 40% for final holdout testing
- Used **SMOTE** to oversample fraud cases in the training data to a 0.5 ratio
- Trained a baseline **LightGBM classifier**
- Performed hyperparameter tuning using `RandomizedSearchCV` (5-fold CV)
- Evaluated performance before and after tuning on both dev and holdout sets

---

## Tools and Libraries

- Python: `pandas`, `scikit-learn`, `imbalanced-learn`, `lightgbm`, `matplotlib`
- Google Colab
- GitHub
- ChatGPT

---

## Results

### Evaluation on Development Set (30% of 60%)

#### Before Hyperparameter Tuning:
| Metric     | Score   |
|------------|---------|
| Accuracy   | 0.9992  |
| Precision  | 0.7353  |
| Recall     | 0.7895  |
| F1 Score   | 0.7614  |

---

#### After Hyperparameter Tuning:
| Metric     | Score   |
|------------|---------|
| Accuracy   | 0.9993  |
| Precision  | 0.7755  |
| Recall     | 0.8539  |
| F1 Score   | 0.8128  |

---

### Evaluation on Holdout Test Set (Final 40%)

| Metric     | Score   |
|------------|---------|
| Accuracy   | 0.9992  |
| Precision  | 0.7397  |
| Recall     | 0.8223  |
| F1 Score   | 0.7788  |

The model showed strong generalization performance on unseen data.

---

##  Model Pipeline

- A separate prediction script (`fraud_pipeline.py`) is included to automate:
  1. Loading new transaction data
  2. Applying preprocessing
  3. Loading the trained model
  4. Making predictions and fraud probability scores
  5. Exporting results to CSV

---

## 📂 Repository Structure

