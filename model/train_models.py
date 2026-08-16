"""
train_models.py
----------------
Script to train and evaluate multiple ML classifiers on the BRFSS 2015 Heart Disease dataset.
Saves trained models, scaler, feature columns, and evaluation metrics for later use in the Streamlit app.
Outputs:
  - test_data.csv (held-out test set with labels)
  - metrics_comparison.csv (summary of model performance)


Dataset (download manually, see README.md "Dataset" section):
    Heart Disease Health Indicators Dataset (BRFSS 2015)
    https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset
    File: heart_disease_health_indicators_BRFSS2015.csv
    Target column: HeartDiseaseorAttack (0 = No, 1 = Yes)

Usage:
    python train_models.py --data ../data/heart_disease_health_indicators_BRFSS2015.csv
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

TARGET_COL = "HeartDiseaseorAttack"
RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV, drop missing values, and validate target column."""
    df = pd.read_csv(path)
    df = df.dropna().reset_index(drop=True)
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )
    return df


def evaluate(model, X_test, y_test, scaled_test=None):
    """Evaluate a trained model on test data and return key classification metrics.""" 
    X_eval = scaled_test if scaled_test is not None else X_test
    y_pred = model.predict(X_eval)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_eval)[:, 1]
    else:
        y_proba = y_pred  # fallback

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main(data_path: str, out_dir: str, test_size: float, sample_frac: float):
    df = load_data(data_path)

    # BRFSS dataset is very large (~250k rows). To speed up training, we can sample a fraction.
    # Use --sample_frac 1.0 if you want to train on the full dataset.

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=RANDOM_STATE).reset_index(drop=True)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

   # Scaling is important for distance-based models (LR, KNN, NB).
   # Decision Tree and Random Forest work fine without scaling, so they use raw features.

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    uses_scaled = {"logistic_regression", "knn", "naive_bayes"}

    os.makedirs(out_dir, exist_ok=True)
    results = {}

    for name, model in models.items():
        if name in uses_scaled:
            model.fit(X_train_scaled, y_train)
            metrics = evaluate(model, X_test, y_test, scaled_test=X_test_scaled)
        else:
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test, scaled_test=None)

        results[name] = metrics
        joblib.dump(model, os.path.join(out_dir, f"{name}.pkl"))
        print(f"[{name}] Metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    # Save scaler and feature column order (needed by the Streamlit app)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(list(X.columns), os.path.join(out_dir, "feature_columns.pkl"))
    joblib.dump(list(uses_scaled), os.path.join(out_dir, "uses_scaled.pkl"))

    # Save evaluation metrics for all models into a CSV (used in README and Streamlit app).

    display_names = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "knn": "kNN",
        "naive_bayes": "Naive Bayes",
        "random_forest": "Random Forest (Ensemble)",
    }
    comp_df = pd.DataFrame(
        [{"ML Model Name": display_names[k], **v} for k, v in results.items()]
    )
    comp_df.to_csv(os.path.join(out_dir, "metrics_comparison.csv"), index=False)
    print("\nSaved metrics_comparison.csv with all model results.")
    print(comp_df.to_string(index=False))

    # Save the held-out test split (features + TRUE label) as test_data.csv
    # at the project root -- this is the file you upload to the Streamlit app
    # and the file required in the submission.
    test_out = X_test.copy()
    test_out[TARGET_COL] = y_test.values
    test_csv_path = os.path.join(os.path.dirname(out_dir.rstrip("/")), "test_data.csv")
    test_out.to_csv(test_csv_path, index=False)
    print(f"\nExported test_data.csv (includes true labels) to {test_csv_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="../data/heart_disease_health_indicators_BRFSS2015.csv",
        help="Path to the raw dataset CSV",
    )
    parser.add_argument(
        "--out_dir", type=str, default=".", help="Directory to save trained models"
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=0.15,
        help="Fraction of the full dataset to use (BRFSS has 250k+ rows; "
        "0.15 keeps training fast while still far exceeding 500 instances). "
        "Use 1.0 for the full dataset.",
    )
    args = parser.parse_args()
    main(args.data, args.out_dir, args.test_size, args.sample_frac)
