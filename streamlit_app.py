# ============================
# 🖥 Streamlit App (v19)
# ============================
import streamlit as st
import pandas as pd
import numpy as np
import os, json, joblib, io

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

def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# ============================
# Streamlit App
# ============================
st.set_page_config(page_title="Parkinson’s Prediction", layout="wide")
st.title("🧠 Parkinson’s Prediction Project (v19)")

tabs = st.tabs(["EDA", "Model Results", "Prediction", "Retrain", "Explainability", "Training Log"])

# -------------------------
# Tab 1: EDA
# -------------------------
with tabs[0]:
    st.header("🔍 Exploratory Data Analysis")
    df = load_data()
    st.write("### נתונים ראשוניים")
    st.dataframe(df.head())

    # Export options
    st.download_button("⬇️ Download Data (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "eda_data.csv", "text/csv")

    with io.BytesIO() as buffer:
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="EDA")
        st.download_button("⬇️ Download Data (Excel)", buffer.getvalue(),
                           "eda_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.write("### גרפים וניתוחים")
    if os.path.exists(ASSETS_DIR):
        image_files = sorted([f for f in os.listdir(ASSETS_DIR) if f.endswith(".png")])
        if image_files:
            for i in range(0, len(image_files), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i+j < len(image_files):
                        img = image_files[i+j]
                        cols[j].image(os.path.join(ASSETS_DIR, img), caption=img)
        else:
            st.info("⚠️ לא נמצאו גרפים בתיקיית assets.")
    else:
        st.error("❌ תיקיית assets לא נמצאה.")

# -------------------------
# Tab 2: Model Results
# -------------------------
with tabs[1]:
    st.header("🤖 Model Results")
    leaderboard = load_leaderboard()

    if leaderboard:
        lb_df = pd.DataFrame({
            model: scores["test"] for model, scores in leaderboard.items()
        }).T.reset_index().rename(columns={"index": "Model"})

        # Rank by ROC-AUC
        lb_df = lb_df.sort_values(by="roc_auc", ascending=False).reset_index(drop=True)
        lb_df.index = lb_df.index + 1
        lb_df.insert(0, "Rank", lb_df.index)

        # Highlight best model
        best_model = lb_df.iloc[0]["Model"]
        lb_df.loc[lb_df["Model"] == best_model, "Model"] = "🏆 " + best_model

        st.write("### Leaderboard – Model Comparison")
        st.dataframe(
            lb_df.style.highlight_max(
                axis=0, subset=["accuracy","precision","recall","f1","roc_auc"], color="lightgreen"
            )
        )

        # Export leaderboard
        st.download_button("⬇️ Download Leaderboard (CSV)", lb_df.to_csv(index=False).encode("utf-8"),
                           "leaderboard.csv", "text/csv")
    else:
        st.warning("⚠️ No leaderboard found. Run pipeline first.")

    for plot in ["roc_curve.png", "pr_curve.png", "learning_curve.png", "confusion_matrix.png"]:
        path = os.path.join(ASSETS_DIR, plot)
        if os.path.exists(path):
            st.image(path, caption=plot)

# -------------------------
# Tab 3: Prediction
# -------------------------
with tabs[2]:
    st.header("🩺 Prediction")
    model = load_model()
    df = load_data()

    if model:
        option = st.radio("Choose input method:", ["Manual Entry", "Upload CSV"])

        if option == "Manual Entry":
            inputs = {col: st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
                      for col in df.drop("status", axis=1).columns}
            if st.button("Predict"):
                X_new = pd.DataFrame([inputs])
                prob = model.predict_proba(X_new)[0, 1]
                label = "Parkinson’s" if prob > 0.5 else "Healthy"

                if prob < 0.3:
                    risk = "🟢 Low Risk"
                elif prob < 0.7:
                    risk = "🟡 Medium Risk"
                else:
                    risk = "🔴 High Risk"

                st.write(f"Prediction: **{label}** (prob={prob:.2f}) {risk}")
                st.progress(int(prob * 100))

        else:
            file = st.file_uploader("Upload CSV", type=["csv"])
            if file:
                new_df = pd.read_csv(file)
                preds = model.predict(new_df)
                probs = model.predict_proba(new_df)[:, 1]

                results = new_df.copy()
                results["prediction"] = preds
                results["probability"] = probs
                results["risk_label"] = ["🟢 Low" if p < 0.3 else "🟡 Medium" if p < 0.7 else "🔴 High" for p in probs]

                st.dataframe(results)

                # Export results
                st.download_button("⬇️ Download Predictions (CSV)",
                                   data=results.to_csv(index=False).encode("utf-8"),
                                   file_name="predictions.csv",
                                   mime="text/csv")

                with io.BytesIO() as buffer:
                    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                        results.to_excel(writer, index=False, sheet_name="Predictions")
                    st.download_button("⬇️ Download Predictions (Excel)", buffer.getvalue(),
                                       "predictions.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -------------------------
# Tab 4: Retrain
# -------------------------
with tabs[3]:
    st.header("🔄 Retrain Models")
    uploaded = st.file_uploader("Upload new dataset (CSV)", type=["csv"])
    if uploaded:
        new_df = pd.read_csv(uploaded)
        new_df.to_csv(os.path.join(DATA_DIR, "parkinsons.csv"), index=False)
        st.success("Dataset updated! Please rerun model_pipeline.py manually to retrain.")

# -------------------------
# Tab 5: Explainability
# -------------------------
with tabs[4]:
    st.header("🧾 Explainability")
    shap_path = os.path.join(ASSETS_DIR, "shap_summary.png")
    if os.path.exists(shap_path):
        st.image(shap_path, caption="SHAP Summary Plot")
    else:
        st.info("No SHAP plot available yet. Run pipeline first.")

# -------------------------
# Tab 6: Training Log
# -------------------------
with tabs[5]:
    st.header("📜 Training Log")
    log_path = os.path.join(ASSETS_DIR, "training_log.csv")
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        st.dataframe(log_df)

        st.download_button("⬇️ Download Log (CSV)", log_df.to_csv(index=False).encode("utf-8"),
                           "training_log.csv", "text/csv")
    else:
        st.info("No training log available yet.")
