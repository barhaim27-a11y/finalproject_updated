# ============================
# 🖥 Streamlit App (v19_pro)
# ============================
import streamlit as st
import pandas as pd
import numpy as np
import os, json, joblib, datetime, io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay, precision_recall_curve, auc
)
from sklearn.model_selection import train_test_split, learning_curve
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
    if "best_model" not in st.session_state:
        if os.path.exists(model_path):
            st.session_state.best_model = joblib.load(model_path)
        else:
            st.session_state.best_model = None
    return st.session_state.best_model

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
        auc_score = roc_auc_score(y, y_prob)
        results[name] = {
            "test": {
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred),
                "recall": recall_score(y, y_pred),
                "f1": f1_score(y, y_pred),
                "roc_auc": auc_score,
            }
        }
        if auc_score > best_auc:
            best_auc, best_model, best_name = auc_score, pipe, name
    return results, best_model, best_name

# ============================
# Export helper
# ============================
def export_download(df, name="export.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, name, "text/csv")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    st.download_button("⬇️ Download Excel", buffer.getvalue(), name.replace(".csv",".xlsx"), "application/vnd.ms-excel")

# ============================
# Streamlit App
# ============================
st.set_page_config(page_title="Parkinson’s Prediction", layout="wide")
st.title("🧠 Parkinson’s Prediction Project (v19_pro)")

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

    st.subheader("📊 Dataset Statistics")
    st.dataframe(df.describe().T)
    export_download(df.describe().reset_index(), "eda_stats.csv")

    # Layout 2 cols for plots
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.countplot(x="status", data=df, palette="Set2", ax=ax)
        ax.set_title("Target Distribution")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(df.corr(), cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

    st.subheader("📈 Pairplot (sampled)")
    sampled = df.sample(min(200, len(df)), random_state=42)
    fig = sns.pairplot(sampled, hue="status", diag_kind="kde", plot_kws={'alpha':0.5})
    st.pyplot(fig)

    st.subheader("📦 Feature Distribution by Status (first 6 features)")
    num_cols = df.drop("status", axis=1).columns[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15,8))
    for i, col in enumerate(num_cols):
        sns.boxplot(x="status", y=col, data=df, ax=axes[i//3, i%3])
    st.pyplot(fig)

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

        best_row = lb_df.iloc[0]
        best_model_name = best_row["Model"]
        st.success(f"🏆 Best Model: **{best_model_name}** (ROC-AUC={best_row['roc_auc']:.3f})")

        st.dataframe(lb_df)
        export_download(lb_df,"leaderboard.csv")

        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(x="roc_auc", y="Model", data=lb_df, palette="Blues_r", ax=ax)
        st.pyplot(fig)

        # Extra curves
        st.subheader("📉 Best Model Evaluation")
        model = load_model()
        df = load_data()
        X, y = df.drop("status", axis=1), df["status"]
        if model:
            y_prob = model.predict_proba(X)[:,1]
            y_pred = model.predict(X)

            # ROC
            fig, ax = plt.subplots()
            RocCurveDisplay.from_predictions(y, y_prob, ax=ax)
            st.pyplot(fig)

            # Precision-Recall
            prec, rec, _ = precision_recall_curve(y, y_prob)
            fig, ax = plt.subplots()
            ax.plot(rec, prec, label="PR Curve")
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            st.pyplot(fig)

            # Learning curve
            train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring="roc_auc")
            fig, ax = plt.subplots()
            ax.plot(train_sizes, train_scores.mean(axis=1), label="Train")
            ax.plot(train_sizes, test_scores.mean(axis=1), label="Test")
            ax.legend(); ax.set_title("Learning Curve")
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
                res = pd.DataFrame([{"probability":prob,"label":label,"risk":risk}])
                st.dataframe(res)
                export_download(res,"prediction.csv")
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
                export_download(results,"batch_predictions.csv")
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
        export_download(log_df,"training_log.csv")
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
            new_leaderboard, new_best_model, new_best_name = run_pipeline(X, y)

        lb_df = pd.DataFrame({m:s["test"] for m,s in new_leaderboard.items()}).T.reset_index().rename(columns={"index":"Model"})
        st.dataframe(lb_df)
        export_download(lb_df,"retrain_results.csv")

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
        res = pd.DataFrame([{
            "accuracy": accuracy_score(y_test,y_pred),
            "precision": precision_score(y_test,y_pred),
            "recall": recall_score(y_test,y_pred),
            "f1": f1_score(y_test,y_pred),
            "roc_auc": roc_auc_score(y_test,y_prob)
        }])
        st.dataframe(res)
        export_download(res,"playground_results.csv")
        fig,ax=plt.subplots(); RocCurveDisplay.from_predictions(y_test,y_prob,ax=ax); st.pyplot(fig)
