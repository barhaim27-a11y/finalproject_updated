# ============================
# 🖥 Streamlit App (v19_final)
# ============================
import streamlit as st
import pandas as pd
import numpy as np
import os, json, joblib, datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay, ConfusionMatrixDisplay
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
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

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
# Retrain Helper
# ============================
def run_pipeline(X, y):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
    }
    results, best_auc, best_model, best_name = {}, -1, None, None
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
            best_auc, best_model, best_name = auc, pipe, name
    return results, best_model, best_name

# ============================
# Streamlit App
# ============================
st.set_page_config(page_title="Parkinson’s Prediction", layout="wide")
st.title("🧠 Parkinson’s Prediction Project (v19_final)")

tabs = st.tabs([
    "EDA", "Model Results", "Prediction", "Explainability", "Training Log", "Retrain", "Playground"
])

# -------------------------
# Tab 1: EDA
# -------------------------
with tabs[0]:
    st.header("🔍 Exploratory Data Analysis")
    df = load_data()
    st.dataframe(df.head())

    # Distribution
    fig, ax = plt.subplots()
    sns.countplot(x="status", data=df, palette="Set2", ax=ax)
    ax.set_title("Target Distribution (0=Healthy, 1=Parkinson’s)")
    st.pyplot(fig)

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(df.corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

    # Pairplot (sampled)
    st.subheader("Pairplot (sampled)")
    sampled = df.sample(min(200, len(df)), random_state=42)
    fig = sns.pairplot(sampled, hue="status", diag_kind="kde", plot_kws={'alpha':0.5})
    st.pyplot(fig)

    # Boxplots
    st.subheader("Feature Distribution by Status")
    num_cols = df.drop("status", axis=1).columns[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15,8))
    for i, col in enumerate(num_cols):
        sns.boxplot(x="status", y=col, data=df, ax=axes[i//3, i%3])
    st.pyplot(fig)

    # Feature Importance (אם קיים asset)
    feat_path = os.path.join(ASSETS_DIR,"feature_importance.png")
    if os.path.exists(feat_path):
        st.subheader("Feature Importance")
        st.image(feat_path)

# -------------------------
# Tab 2: Model Results
# -------------------------
with tabs[1]:
    st.header("🤖 Model Results")
    leaderboard = load_leaderboard()
    if leaderboard:
        lb_df = pd.DataFrame({m: s["test"] for m,s in leaderboard.items()}).T.reset_index().rename(columns={"index":"Model"})
        lb_df = lb_df.sort_values(by="roc_auc", ascending=False).reset_index(drop=True)
        lb_df.index = lb_df.index + 1
        lb_df["Rank"] = lb_df.index

        # Highlight best model
        best_row = lb_df.iloc[0]
        best_model = best_row["Model"]
        st.success(f"🏆 Best Model: **{best_model}** (ROC-AUC={best_row['roc_auc']:.3f})")

        # Show table
        st.dataframe(lb_df[["Rank","Model","accuracy","precision","recall","f1","roc_auc"]])

        # Barplot
        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(x="roc_auc", y="Model", data=lb_df, palette="Blues_r", ax=ax)
        for i, v in enumerate(lb_df["roc_auc"]):
            ax.text(v+0.005, i, f"{v:.2f}", va="center")
        ax.set_title("Model Comparison (ROC AUC)")
        st.pyplot(fig)
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
                risk = "🟢 Low" if prob<0.3 else ("🟡 Medium" if prob<0.7 else "🔴 High")
                st.progress(prob)
                st.write(f"Prediction: **{label}** ({prob:.2f}), Risk={risk}")
        else:
            file = st.file_uploader("Upload CSV", type=["csv"])
            if file:
                new_df = pd.read_csv(file)
                preds = model.predict(new_df)
                probs = model.predict_proba(new_df)[:,1]
                results = new_df.copy()
                results["prediction"] = preds
                results["probability"] = probs
                results["risk_label"] = pd.cut(probs, bins=[0,0.3,0.7,1], labels=["🟢 Low","🟡 Medium","🔴 High"])
                st.dataframe(results)
                st.download_button("⬇️ Download Predictions CSV", results.to_csv(index=False).encode("utf-8"),
                                   "predictions.csv","text/csv")
    else:
        st.error("No trained model found.")

# -------------------------
# Tab 4: Explainability
# -------------------------
with tabs[3]:
    st.header("🧾 Explainability")
    shap_path = os.path.join(ASSETS_DIR,"shap_summary.png")
    if os.path.exists(shap_path):
        st.image(shap_path)
    else:
        st.info("No SHAP results yet.")

# -------------------------
# Tab 5: Training Log
# -------------------------
with tabs[4]:
    st.header("📜 Training Log")
    log_path = os.path.join(ASSETS_DIR,"training_log.csv")
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        st.dataframe(log_df)
        st.download_button("⬇️ Download Log CSV", log_df.to_csv(index=False).encode("utf-8"),
                           "training_log.csv","text/csv")
    else:
        st.info("No log yet.")

# -------------------------
# Tab 6: Retrain (דינאמי)
# -------------------------
with tabs[5]:
    st.header("🔄 Retrain Models")
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
                new_leaderboard, new_best_model, new_best_name = run_pipeline(X, y)

        if 'new_leaderboard' in locals():
            lb_df = pd.DataFrame({m:s["test"] for m,s in new_leaderboard.items()}).T.reset_index().rename(columns={"index":"Model"})
            st.dataframe(lb_df)
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(new_best_model,X,y,ax=ax)
            st.pyplot(fig)

            if st.button("✅ Promote New Model as Best"):
                joblib.dump(new_best_model, model_path)
                with open(leaderboard_path,"w") as f: json.dump(new_leaderboard,f,indent=4)
                combined_df.to_csv(os.path.join(DATA_DIR,"parkinsons.csv"),index=False)
                log_path = os.path.join(ASSETS_DIR,"training_log.csv")
                new_row = pd.DataFrame([{"date":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                         "dataset_size":combined_df.shape[0],
                                         "best_model":new_best_name}])
                if os.path.exists(log_path):
                    log_df = pd.read_csv(log_path); log_df = pd.concat([log_df,new_row],ignore_index=True)
                else:
                    log_df = new_row
                log_df.to_csv(log_path,index=False)
                st.success(f"🎉 {new_best_name} promoted!")

# -------------------------
# Tab 7: Playground
# -------------------------
with tabs[6]:
    st.header("🛠 Model Playground")
    df = load_data(); X = df.drop("status",axis=1); y = df["status"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    choice = st.selectbox("Choose model",["Random Forest","Logistic Regression","XGBoost","LightGBM"])
    if st.button("Train"):
        if choice=="Random Forest": model=RandomForestClassifier(n_estimators=200)
        elif choice=="Logistic Regression": model=LogisticRegression(max_iter=500)
        elif choice=="XGBoost": model=XGBClassifier(use_label_encoder=False,eval_metric="logloss")
        else: model=LGBMClassifier()
        pipe=Pipeline([("scaler",StandardScaler()),("clf",model)])
        pipe.fit(X_train,y_train)
        y_pred=pipe.predict(X_test); y_prob=pipe.predict_proba(X_test)[:,1]
        st.write("Accuracy:",accuracy_score(y_test,y_pred))
        fig,ax=plt.subplots(); RocCurveDisplay.from_predictions(y_test,y_prob,ax=ax); st.pyplot(fig)
