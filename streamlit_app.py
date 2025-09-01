# ============================
# 🖥 Streamlit App (v17)
# ============================
import streamlit as st
import pandas as pd
import numpy as np
import os, json, joblib

# Paths
ASSETS_DIR = "assets"
MODELS_DIR = "models"
DATA_DIR = "data"

metrics_path = os.path.join(ASSETS_DIR,"metrics.json")
leaderboard_path = os.path.join(ASSETS_DIR,"leaderboard.json")
model_path = os.path.join(MODELS_DIR,"best_model.joblib")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR,"parkinsons.csv"))
    return df

# Load metrics
def load_metrics():
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return {}

def load_leaderboard():
    if os.path.exists(leaderboard_path):
        with open(leaderboard_path) as f:
            return json.load(f)
    return {}

# Load model
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

st.set_page_config(page_title="Parkinson’s Prediction", layout="wide")

st.title("🧠 Parkinson’s Prediction Project (v17)")

# Tabs
tabs = st.tabs(["EDA", "Model Results", "Prediction", "Retrain", "Explainability", "Training Log"])

# -------------------------
# Tab 1: EDA
# -------------------------
with tabs[0]:
    st.header("🔍 Exploratory Data Analysis")
    df = load_data()
    st.write(df.head())
    if os.path.exists(os.path.join(ASSETS_DIR,"correlation_heatmap.png")):
        st.image(os.path.join(ASSETS_DIR,"correlation_heatmap.png"), caption="Correlation Heatmap")
    if os.path.exists(os.path.join(ASSETS_DIR,"pca.png")):
        st.image(os.path.join(ASSETS_DIR,"pca.png"), caption="PCA Projection")
    if os.path.exists(os.path.join(ASSETS_DIR,"tsne.png")):
        st.image(os.path.join(ASSETS_DIR,"tsne.png"), caption="t-SNE Projection")

# -------------------------
# Tab 2: Model Results
# -------------------------
with tabs[1]:
    st.header("🤖 Model Results")
    metrics = load_metrics()
    leaderboard = load_leaderboard()
    if metrics:
        cols = st.columns(5)
        cols[0].metric("Accuracy", f"{metrics['test']['accuracy']:.2f}")
        cols[1].metric("Precision", f"{metrics['test']['precision']:.2f}")
        cols[2].metric("Recall", f"{metrics['test']['recall']:.2f}")
        cols[3].metric("F1", f"{metrics['test']['f1']:.2f}")
        cols[4].metric("ROC-AUC", f"{metrics['test']['roc_auc']:.2f}")
    if leaderboard:
        st.write("### Leaderboard")
        lb_df = pd.DataFrame({k:v['test'] for k,v in leaderboard.items()}).T
        st.dataframe(lb_df)
    for plot in ["roc_curve.png","pr_curve.png","learning_curve.png","confusion_matrix.png"]:
        if os.path.exists(os.path.join(ASSETS_DIR,plot)):
            st.image(os.path.join(ASSETS_DIR,plot), caption=plot)

# -------------------------
# Tab 3: Prediction
# -------------------------
with tabs[2]:
    st.header("🩺 Prediction")
    model = load_model()
    if model:
        option = st.radio("Choose input method:", ["Manual Entry","Upload CSV"])
        if option=="Manual Entry":
            inputs = {col: st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean())) for col in df.drop("status",axis=1).columns}
            if st.button("Predict"):
                X_new = pd.DataFrame([inputs])
                prob = model.predict_proba(X_new)[0,1]
                label = "Parkinson’s" if prob>0.5 else "Healthy"
                st.write(f"Prediction: **{label}** (prob={prob:.2f})")
        else:
            file = st.file_uploader("Upload CSV", type=["csv"])
            if file:
                new_df = pd.read_csv(file)
                preds = model.predict(new_df)
                st.write("Predictions:", preds)

# -------------------------
# Tab 4: Retrain
# -------------------------
with tabs[3]:
    st.header("🔄 Retrain Models")
    uploaded = st.file_uploader("Upload new dataset (CSV)", type=["csv"])
    if uploaded:
        new_df = pd.read_csv(uploaded)
        new_df.to_csv(os.path.join(DATA_DIR,"parkinsons.csv"), index=False)
        st.success("Dataset updated! Please rerun model_pipeline.py manually to retrain.")

# -------------------------
# Tab 5: Explainability
# -------------------------
with tabs[4]:
    st.header("🧾 Explainability")
    if os.path.exists(os.path.join(ASSETS_DIR,"shap_summary.png")):
        st.image(os.path.join(ASSETS_DIR,"shap_summary.png"), caption="SHAP Summary Plot")
    else:
        st.info("No SHAP plot available yet. Run pipeline first.")

# -------------------------
# Tab 6: Training Log
# -------------------------
with tabs[5]:
    st.header("📜 Training Log")
    log_path = os.path.join(ASSETS_DIR,"training_log.csv")
    if os.path.exists(log_path):
        st.dataframe(pd.read_csv(log_path))
    else:
        st.info("No training log available yet.")
