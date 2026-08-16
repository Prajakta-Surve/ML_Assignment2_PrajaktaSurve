"""
app.py
------

Streamlit dashboard for Assignment 2 (Machine learning).
Provides:
  - Upload option for test dataset (CSV)
  - Dropdown to select trained model
  - Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
  - Confusion matrix and classification report
  - Comparison table across all models


"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_DIR = "model"
TARGET_COL = "HeartDiseaseorAttack"

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "kNN",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest (Ensemble)",
}

st.set_page_config(page_title="Heart Disease Classifier", layout="wide")


@st.cache_resource
def load_artifacts():
    models = {}
    for key in DISPLAY_NAMES:
        path = os.path.join(MODEL_DIR, f"{key}.pkl")
        if os.path.exists(path):
            models[key] = joblib.load(path)

    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    cols_path = os.path.join(MODEL_DIR, "feature_columns.pkl")
    uses_scaled_path = os.path.join(MODEL_DIR, "uses_scaled.pkl")

    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    feature_columns = joblib.load(cols_path) if os.path.exists(cols_path) else None
    uses_scaled = joblib.load(uses_scaled_path) if os.path.exists(uses_scaled_path) else set()

    return models, scaler, feature_columns, uses_scaled


def main():
    st.title("❤️ Heart Disease Prediction — Model Comparison Dashboard")
    st.caption(
        "Assignment 2 | Machine Learning | Dataset: Heart Disease Health "
        "Indicators (BRFSS 2015)"
    )

    models, scaler, feature_columns, uses_scaled = load_artifacts()

    if not models:
        st.error(
            "No trained models found in the `model/` folder. Run "
            "`python model/train_models.py` first to generate them."
        )
        st.stop()

    # ---------------- Sidebar: model selection ----------------
    st.sidebar.header("⚙️ Configuration")
    model_key = st.sidebar.selectbox(
        "Select a model",
        options=list(models.keys()),
        format_func=lambda k: DISPLAY_NAMES[k],
    )
    model = models[model_key]

    # ---------------- Dataset upload ----------------
    st.sidebar.header("📁 Upload Test Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload test_data.csv (must include the true label column "
        f"'{TARGET_COL}')",
        type=["csv"],
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded uploaded file with {df.shape[0]} rows, {df.shape[1]} columns.")
    elif os.path.exists("test_data.csv"):
        df = pd.read_csv("test_data.csv")
        st.info("No file uploaded — using bundled `test_data.csv` by default.")
    else:
        st.warning("Please upload a test CSV file to continue.")
        st.stop()

    st.subheader("🔍 Preview of Test Data")
    st.dataframe(df.head(10), use_container_width=True)

    if TARGET_COL not in df.columns:
        st.error(
            f"Uploaded CSV must contain the true label column '{TARGET_COL}' "
            "to compute evaluation metrics."
        )
        st.stop()

    if feature_columns is None:
        st.error("feature_columns.pkl not found in model/. Re-run training script.")
        st.stop()

    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing required feature columns: {missing_cols}")
        st.stop()

    X = df[feature_columns]
    y_true = df[TARGET_COL].astype(int)

    # Scale if this model needs scaled input
    if model_key in uses_scaled and scaler is not None:
        X_input = scaler.transform(X)
    else:
        X_input = X

    y_pred = model.predict(X_input)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_input)[:, 1]
    else:
        y_proba = y_pred

    # ---------------- Metrics ----------------
    st.subheader(f"📊 Evaluation Metrics — {DISPLAY_NAMES[model_key]}")

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    cols = st.columns(len(metrics))
    for c, (name, value) in zip(cols, metrics.items()):
        c.metric(name, f"{value:.3f}")

    # ---------------- Confusion Matrix + Classification Report ----------------
    left, right = st.columns(2)

    with left:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with right:
        st.markdown("**Classification Report**")
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    # ---------------- Model Comparison (if metrics_comparison.csv exists) ----------------
    comp_path = os.path.join(MODEL_DIR, "metrics_comparison.csv")
    if os.path.exists(comp_path):
        st.subheader("📈 All Models — Comparison Table")
        comp_df = pd.read_csv(comp_path)
        st.dataframe(comp_df, use_container_width=True)


if __name__ == "__main__":
    main()
