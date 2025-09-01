# ============================
# 🖥 Streamlit App (v20)
# ============================
import streamlit as st
import pandas as pd
import numpy as np
import os, json, joblib, io
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
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
# Streamlit App
# ============================
st.set_page_config(page_title="Parkinson’s Prediction", layout="wide")
st.title("🧠 Parkinson’s Prediction Project (v20)")

tabs = st.tabs([
    "EDA", "Model Results", "Prediction", "Retrain", "Explainability", "Training Log",
    "Model Playground", "Model Comparison Lab"
])

# -------------------------
# Tab 7: Model Playground
# -------------------------
with tabs[6]:
    st.header("🛠 Model Playground")
    df = load_data()
    X = df.drop("status", axis=1)
    y = df["status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model_choice = st.selectbox("בחר מודל:", ["Random Forest", "XGBoost", "Logistic Regression", "LightGBM"])

    params = {}
    if model_choice == "Random Forest":
        params["n_estimators"] = st.slider("n_estimators", 50, 500, 200, 50)
        params["max_depth"] = st.slider("max_depth", 2, 20, 5)
        model = RandomForestClassifier(**params, random_state=42)

    elif model_choice == "XGBoost":
        params["learning_rate"] = st.slider("learning_rate", 0.01, 0.3, 0.1)
        params["n_estimators"] = st.slider("n_estimators", 50, 500, 200, 50)
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, **params)

    elif model_choice == "Logistic Regression":
        params["C"] = st.slider("C (inverse regularization)", 0.01, 5.0, 1.0)
        model = LogisticRegression(max_iter=500, **params)

    elif model_choice == "LightGBM":
        params["n_estimators"] = st.slider("n_estimators", 50, 500, 200, 50)
        params["learning_rate"] = st.slider("learning_rate", 0.01, 0.3, 0.1)
        model = LGBMClassifier(random_state=42, **params)

    if st.button("🚀 Train Model"):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }

        st.subheader("📊 Results")
        st.write(pd.DataFrame([results]))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
        st.pyplot(fig)

        # Export results
        res_df = pd.DataFrame([{"Model": model_choice, **params, **results}])
        st.download_button("⬇️ Save Results (CSV)", res_df.to_csv(index=False).encode("utf-8"),
                           "playground_results.csv", "text/csv")

        with io.BytesIO() as buffer:
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                res_df.to_excel(writer, index=False, sheet_name="Playground")
            st.download_button("⬇️ Save Results (Excel)", buffer.getvalue(),
                               "playground_results.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -------------------------
# Tab 8: Model Comparison Lab
# -------------------------
with tabs[7]:
    st.header("⚖️ Model Comparison Lab")
    leaderboard = load_leaderboard()
    df = load_data()
    X = df.drop("status", axis=1)
    y = df["status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if leaderboard:
        models_available = list(leaderboard.keys())
        selected = st.multiselect("בחר מודלים להשוואה", models_available, default=models_available[:2])

        if st.button("🔍 Compare"):
            comp_results = []
            fig, ax = plt.subplots()

            for m in selected:
                try:
                    model_file = os.path.join(MODELS_DIR, f"{m.replace(' ', '_')}.joblib")
                    if os.path.exists(model_file):
                        mdl = joblib.load(model_file)
                    else:
                        mdl = load_model()  # fallback to best_model

                    y_pred = mdl.predict(X_test)
                    y_prob = mdl.predict_proba(X_test)[:, 1]

                    comp_results.append({
                        "Model": m,
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "Precision": precision_score(y_test, y_pred),
                        "Recall": recall_score(y_test, y_pred),
                        "F1": f1_score(y_test, y_pred),
                        "ROC_AUC": roc_auc_score(y_test, y_prob)
                    })

                    RocCurveDisplay.from_predictions(y_test, y_prob, name=m, ax=ax)
                except Exception as e:
                    st.error(f"Model {m} failed: {e}")

            if comp_results:
                comp_df = pd.DataFrame(comp_results).sort_values(by="ROC_AUC", ascending=False)
                st.subheader("📊 Comparison Results")
                st.dataframe(comp_df)

                st.subheader("📈 ROC Curves")
                st.pyplot(fig)

                # Export
                st.download_button("⬇️ Save Comparison (CSV)", comp_df.to_csv(index=False).encode("utf-8"),
                                   "comparison_results.csv", "text/csv")

                with io.BytesIO() as buffer:
                    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                        comp_df.to_excel(writer, index=False, sheet_name="Comparison")
                    st.download_button("⬇️ Save Comparison (Excel)", buffer.getvalue(),
                                       "comparison_results.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.warning("⚠️ No leaderboard found. Run pipeline first.")
