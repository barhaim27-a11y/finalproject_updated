# ============================
# 🖥 Streamlit App (v21)
# ============================
import streamlit as st
import pandas as pd
import numpy as np
import os, json, joblib, io, datetime
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Paths
ASSETS_DIR = "assets"
MODELS_DIR = "models"
DATA_DIR = "data"

metrics_path = os.path.join(ASSETS_DIR, "metrics.json")
leaderboard_path = os.path.join(ASSETS_DIR, "leaderboard.json")
model_path = os.path.join(MODELS_DIR, "best_model.joblib")

# ============================
# Loaders
# ============================
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(DATA_DIR, "parkinsons.csv"))

def load_leaderboard():
    if os.path.exists(leaderboard_path):
        with open(leaderboard_path) as f:
            return json.load(f)
    return {}

def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# ============================
# Helpers for Retrain
# ============================
def run_pipeline(X, y):
    """ Train standard pipeline with fixed models """
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
    }
    results = {}
    best_auc = -1
    best_model = None
    best_name = None

    for name, model in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
        pipe.fit(X, y)
        y_pred = pipe.predict(X)
        y_prob = pipe.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_prob)
        results[name] = {
            "test": {
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred),
                "recall": recall_score(y, y_pred),
                "f1": f1_score(y, y_pred),
                "roc_auc": auc,
            }
        }
        if auc > best_auc:
            best_auc = auc
            best_model = pipe
            best_name = name
    return results, best_model, best_name

def run_custom(X, y, selected):
    """ Train only selected models """
    model_map = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
    }
    results = {}
    best_auc = -1
    best_model = None
    best_name = None

    for name in selected:
        model = model_map[name]
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
        pipe.fit(X, y)
        y_pred = pipe.predict(X)
        y_prob = pipe.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_prob)
        results[name] = {
            "test": {
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred),
                "recall": recall_score(y, y_pred),
                "f1": f1_score(y, y_pred),
                "roc_auc": auc,
            }
        }
        if auc > best_auc:
            best_auc = auc
            best_model = pipe
            best_name = name
    return results, best_model, best_name

# ============================
# Streamlit App
# ============================
st.set_page_config(page_title="Parkinson’s Prediction", layout="wide")
st.title("🧠 Parkinson’s Prediction Project (v21)")

tabs = st.tabs([
    "EDA", "Model Results", "Prediction", "Retrain", "Explainability",
    "Training Log", "Model Playground", "Model Comparison Lab"
])

# -------------------------
# Tab 4: Retrain (v21)
# -------------------------
with tabs[3]:
    st.header("🔄 Retrain Models (v21)")
    option = st.radio("Training Mode", ["Pipeline (all models)", "Custom"])

    uploaded = st.file_uploader("Upload new dataset (CSV)", type=["csv"])
    if uploaded:
        new_df = pd.read_csv(uploaded)
        base_df = load_data()
        combined_df = pd.concat([base_df, new_df], ignore_index=True)
        X = combined_df.drop("status", axis=1)
        y = combined_df["status"]

        if option == "Pipeline (all models)":
            st.info("מריץ את כל המודלים מה־pipeline הקיים...")
            new_leaderboard, new_best_model, new_best_name = run_pipeline(X, y)

        elif option == "Custom":
            st.info("בחר מודלים מותאמים לאימון:")
            selected = st.multiselect("Select models", ["Random Forest", "Logistic Regression", "XGBoost", "LightGBM"])
            if st.button("🚀 Run Custom Training"):
                new_leaderboard, new_best_model, new_best_name = run_custom(X, y, selected)

        if 'new_leaderboard' in locals():
            st.subheader("📊 Comparison to Current Best")
            current_model = load_model()
            if current_model:
                st.write("✔️ Current Best Model loaded.")
            else:
                st.warning("⚠️ No current model found.")

            # Leaderboard display
            lb_df = pd.DataFrame({
                m: scores["test"] for m, scores in new_leaderboard.items()
            }).T.reset_index().rename(columns={"index":"Model"})
            st.dataframe(lb_df)

            # ROC curves
            fig, ax = plt.subplots()
            for m, scores in new_leaderboard.items():
                try:
                    y_pred = new_best_model.predict(X)
                    y_prob = new_best_model.predict_proba(X)[:, 1]
                    RocCurveDisplay.from_predictions(y, y_prob, name=new_best_name, ax=ax)
                except:
                    pass
            st.pyplot(fig)

            # Promote button
            if st.button("✅ Promote New Model as Best"):
                os.makedirs(MODELS_DIR, exist_ok=True)
                joblib.dump(new_best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
                with open(leaderboard_path, "w") as f:
                    json.dump(new_leaderboard, f, indent=4)

                combined_df.to_csv(os.path.join(DATA_DIR, "parkinsons.csv"), index=False)

                # Update training log
                log_path = os.path.join(ASSETS_DIR,"training_log.csv")
                new_row = pd.DataFrame([{
                    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "dataset_size": combined_df.shape[0],
                    "best_model": new_best_name
                }])
                if os.path.exists(log_path):
                    log_df = pd.read_csv(log_path)
                    log_df = pd.concat([log_df,new_row],ignore_index=True)
                else:
                    log_df = new_row
                log_df.to_csv(log_path,index=False)

                st.success(f"🎉 {new_best_name} הוגדר כמודל הטוב ביותר החדש!")
