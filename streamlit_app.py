# ============================
# 🖥 Streamlit App (v21 – Full)
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

# ============================
# Paths
# ============================
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
# Tab 1: EDA
# -------------------------
with tabs[0]:
    st.header("🔍 Exploratory Data Analysis")
    df = load_data()
    st.dataframe(df.head())

    # Images from assets
    if os.path.exists(ASSETS_DIR):
        imgs = [f for f in os.listdir(ASSETS_DIR) if f.endswith(".png")]
        if imgs:
            for img in imgs:
                st.image(os.path.join(ASSETS_DIR, img), caption=img)
        else:
            st.info("No EDA plots found.")
    else:
        st.warning("Assets folder not found.")

# -------------------------
# Tab 2: Model Results
# -------------------------
with tabs[1]:
    st.header("🤖 Model Results")
    leaderboard = load_leaderboard()
    if leaderboard:
        lb_df = pd.DataFrame({
            m: scores["test"] for m, scores in leaderboard.items()
        }).T.reset_index().rename(columns={"index":"Model"})
        lb_df = lb_df.sort_values(by="roc_auc", ascending=False).reset_index(drop=True)
        lb_df.index = lb_df.index + 1
        st.dataframe(lb_df)
    else:
        st.warning("No leaderboard found.")

# -------------------------
# Tab 3: Prediction
# -------------------------
with tabs[2]:
    st.header("🩺 Prediction")
    model = load_model()
    df = load_data()
    if model:
        option = st.radio("Choose input:", ["Manual", "CSV Upload"])
        if option == "Manual":
            inputs = {c: st.number_input(c, float(df[c].min()), float(df[c].max()), float(df[c].mean())) for c in df.drop("status",axis=1).columns}
            if st.button("Predict"):
                X_new = pd.DataFrame([inputs])
                prob = model.predict_proba(X_new)[0,1]
                label = "Parkinson’s" if prob > 0.5 else "Healthy"
                st.write(f"Prediction: {label}, Prob={prob:.2f}")
        else:
            file = st.file_uploader("Upload CSV", type=["csv"])
            if file:
                new_df = pd.read_csv(file)
                preds = model.predict(new_df)
                probs = model.predict_proba(new_df)[:,1]
                results = new_df.copy()
                results["prediction"] = preds
                results["probability"] = probs
                st.dataframe(results)
    else:
        st.error("No trained model found.")

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
            new_leaderboard, new_best_model, new_best_name = run_pipeline(X, y)
        else:
            selected = st.multiselect("Select models", ["Random Forest", "Logistic Regression", "XGBoost", "LightGBM"])
            if st.button("🚀 Run Custom Training"):
                new_leaderboard, new_best_model, new_best_name = run_custom(X, y, selected)

        if 'new_leaderboard' in locals():
            st.subheader("📊 New Leaderboard")
            lb_df = pd.DataFrame({
                m: scores["test"] for m, scores in new_leaderboard.items()
            }).T.reset_index().rename(columns={"index":"Model"})
            st.dataframe(lb_df)

            if st.button("✅ Promote New Model as Best"):
                joblib.dump(new_best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
                with open(leaderboard_path, "w") as f:
                    json.dump(new_leaderboard, f, indent=4)
                combined_df.to_csv(os.path.join(DATA_DIR, "parkinsons.csv"), index=False)
                st.success(f"{new_best_name} promoted as Best Model!")

# -------------------------
# Tab 5: Explainability
# -------------------------
with tabs[4]:
    st.header("🧾 Explainability")
    shap_path = os.path.join(ASSETS_DIR, "shap_summary.png")
    if os.path.exists(shap_path):
        st.image(shap_path, caption="SHAP Summary")
    else:
        st.info("No SHAP results yet.")

# -------------------------
# Tab 6: Training Log
# -------------------------
with tabs[5]:
    st.header("📜 Training Log")
    log_path = os.path.join(ASSETS_DIR, "training_log.csv")
    if os.path.exists(log_path):
        st.dataframe(pd.read_csv(log_path))
    else:
        st.info("No training log yet.")

# -------------------------
# Tab 7: Model Playground
# -------------------------
with tabs[6]:
    st.header("🛠 Model Playground")
    df = load_data()
    X = df.drop("status",axis=1); y = df["status"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model_choice = st.selectbox("Choose model", ["Random Forest","Logistic Regression","XGBoost","LightGBM"])
    if st.button("Train"):
        if model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=200)
        elif model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=500)
        elif model_choice == "XGBoost":
            model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        else:
            model = LGBMClassifier()
        pipe = Pipeline([("scaler",StandardScaler()),("clf",model)])
        pipe.fit(X_train,y_train)
        y_pred = pipe.predict(X_test)
        st.write("Accuracy:", accuracy_score(y_test,y_pred))

# -------------------------
# Tab 8: Model Comparison Lab
# -------------------------
with tabs[7]:
    st.header("⚖️ Model Comparison Lab")
    leaderboard = load_leaderboard()
    if leaderboard:
        models_available = list(leaderboard.keys())
        selected = st.multiselect("Select models", models_available, default=models_available[:2])
        if st.button("Compare"):
            comp_results = []
            df = load_data()
            X = df.drop("status",axis=1); y = df["status"]
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
            fig, ax = plt.subplots()
            for m in selected:
                try:
                    mdl = load_model()  # for simplicity, re-use best model
                    y_pred = mdl.predict(X_test)
                    y_prob = mdl.predict_proba(X_test)[:,1]
                    comp_results.append({
                        "Model": m,
                        "Accuracy": accuracy_score(y_test,y_pred),
                        "ROC_AUC": roc_auc_score(y_test,y_prob)
                    })
                    RocCurveDisplay.from_predictions(y_test,y_prob,name=m,ax=ax)
                except Exception as e:
                    st.error(f"{m} failed: {e}")
            st.dataframe(pd.DataFrame(comp_results))
            st.pyplot(fig)
    else:
        st.warning("No leaderboard available.")
