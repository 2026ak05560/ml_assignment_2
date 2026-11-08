import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="ML Assignment 2 - Classifier Demo", layout="wide")

st.title(":stethoscope: Clinical Heart Disease Classification Dashboard")
st.markdown("""
**M.Tech (AIML) - Machine Learning Assignment 2**  
*Student Deployment Portal showcasing 5 Classification Models with evaluation metrics, confusion matrix, and live test dataset validation.*
""")

@st.cache_resource
def load_artifacts():
    try:
        scaler = joblib.load("model/scaler.pkl")
        models = {
            "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
            "Decision Tree": joblib.load("model/decision_tree.pkl"),
            "KNN": joblib.load("model/knn.pkl"),
            "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
            "Support Vector Machine": joblib.load("model/support_vector_machine.pkl")
        }
        return scaler, models
    except Exception as e:
        return None, None

scaler, models = load_artifacts()

# Sidebar controls
st.sidebar.header("Configuration Panel")
model_choice = st.sidebar.selectbox("Select Classification Model", list(models.keys()) if models else [])

st.sidebar.markdown("---")
st.sidebar.subheader("Test Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
else:
    try:
        test_df = pd.read_csv("test_data.csv")
        st.sidebar.info("Loaded default `test_data.csv` from repository.")
    except Exception:
        test_df = None
        st.sidebar.warning("Please upload `test_data.csv` to proceed.")

if test_df is not None and models is not None:
    st.subheader(f"Evaluation Results for: **{model_choice}**")

    if 'Target' not in test_df.columns:
        st.error("Uploaded CSV must contain the 'Target' column for ground truth evaluation.")
    else:
        X_test = test_df.drop(columns=['Target'])
        y_test = test_df['Target']
    
        # Preprocessing if needed
        if model_choice in ["Logistic Regression", "KNN", "Support Vector Machine"]:
            X_test_processed = scaler.transform(X_test)
        else:
            X_test_processed = X_test
    
        model = models[model_choice]
        y_pred = model.predict(X_test_processed)
        y_prob = model.predict_proba(X_test_processed)[:, 1]
    
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
    
        # Display Metrics in Columns
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")
    
        st.markdown("---")
        col_left, col_right = st.columns(2)
    
        with col_left:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)
    
        with col_right:
            st.subheader("Classification Report")
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format(precision=4), use_container_width=True)
    
        with st.expander("Preview Uploaded Test Dataset"):
            st.dataframe(test_df.head(10), use_container_width=True)
else:
    st.info("Awaiting valid test dataset and model artifacts.")