import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, roc_auc_score
)

# ============================
# 1. טעינת מודל ו-Scaler
# ============================
model = joblib.load("models/best_model.joblib")
scaler = joblib.load("models/scaler.joblib")

# ============================
# 2. Prediction
# ============================
def predict_with_risk(model, scaler, samples):
    samples_scaled = scaler.transform(samples)
    preds = model.predict(samples_scaled)
    probs = model.predict_proba(samples_scaled)[:, 1]

    mapping = {0: "Healthy", 1: "Parkinson’s"}

    def risk_label(p):
        if p < 0.33:
            return "🟢 Low"
        elif p < 0.66:
            return "🟡 Medium"
        else:
            return "🔴 High"

    results = pd.DataFrame({
        "Prediction": [mapping[p] for p in preds],
        "Probability": probs.round(3),
        "Risk": [risk_label(p) for p in probs]
    }, index=samples.index)

    return results, preds, probs

# ============================
# 3. Model Evaluation Curves
# ============================
def plot_model_curves(model, X, y_true):
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_true, probs)
    fig, ax = plt.subplots()
    ax.plot(rec, prec, label="PR curve")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    st.pyplot(fig)

    # Confusion Matrix
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Parkinson’s"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# ============================
# 4. EDA
# ============================
def run_eda(df):
    st.subheader("🔍 Exploratory Data Analysis (EDA)")

    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Target Distribution", "📊 Histograms", "📦 Boxplots", "🔥 Correlation"])

    # 1. Target Distribution
    with tab1:
        if "status" in df.columns:
            fig, ax = plt.subplots()
            df["status"].value_counts().plot(kind="bar", ax=ax, color=["green", "red"])
            ax.set_title("Distribution of Target (status)")
            ax.set_xticklabels(["Healthy", "Parkinson’s"], rotation=0)
            st.pyplot(fig)
        else:
            st.info("⚠️ אין עמודת 'status' בקובץ – לא ניתן להציג התפלגות יעד")

    # 2. Histograms
    with tab2:
        st.write("התפלגות פיצ'רים עיקריים")
        numeric_cols = df.select_dtypes(include=np.number).columns[:6]
        fig, axes = plt.subplots(2, 3, figsize=(12,6))
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            sns.histplot(df[col], kde=True, ax=axes[i], color="skyblue")
            axes[i].set_title(col)
        plt.tight_layout()
        st.pyplot(fig)

    # 3. Boxplots
    with tab3:
        if "status" in df.columns:
            fig, axes = plt.subplots(2, 3, figsize=(12,6))
            axes = axes.flatten()
            numeric_cols = df.select_dtypes(include=np.number).columns[:6]
            for i, col in enumerate(numeric_cols):
                sns.boxplot(x="status", y=col, data=df, ax=axes[i])
                axes[i].set_title(f"{col} by Status")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("⚠️ אין עמודת 'status' בקובץ – לא ניתן להציג Boxplots")

    # 4. Correlation Heatmap
    with tab4:
        fig, ax = plt.subplots(figsize=(10,8))
        corr = df.corr()
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

# ============================
# 5. Playground
# ============================
def playground_ui(df):
    st.subheader("🎮 Model Playground")

    if "status" not in df.columns:
        st.warning("⚠️ חסרה עמודת 'status' – אי אפשר להריץ Playground")
        return

    X = df.drop(columns=["status"])
    y = df["status"]
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    mode = st.radio("בחר מצב:", ["🔘 Single Model", "📊 Compare Models"])

    if mode == "🔘 Single Model":
        model_choice = st.selectbox("בחר מודל:", ["RandomForest", "XGB", "LGBM", "CatBoost"])

        if model_choice == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier
            n_estimators = st.slider("n_estimators", 50, 500, 100, step=50)
            max_depth = st.slider("max_depth", 2, 20, 5)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        elif model_choice == "XGB":
            from xgboost import XGBClassifier
            n_estimators = st.slider("n_estimators", 50, 500, 100, step=50)
            learning_rate = st.slider("learning_rate", 0.01, 0.5, 0.1)
            model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, eval_metric="logloss", random_state=42)

        elif model_choice == "LGBM":
            from lightgbm import LGBMClassifier
            n_estimators = st.slider("n_estimators", 50, 500, 100, step=50)
            learning_rate = st.slider("learning_rate", 0.01, 0.5, 0.1)
            model = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)

        elif model_choice == "CatBoost":
            from catboost import CatBoostClassifier
            depth = st.slider("depth", 2, 10, 6)
            iterations = st.slider("iterations", 50, 500, 100, step=50)
            model = CatBoostClassifier(depth=depth, iterations=iterations, verbose=0, random_state=42)

        if st.button("🚀 הרץ מודל"):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            st.write(f"📊 Accuracy: {acc:.3f}, AUC: {auc:.3f}")
            plot_model_curves(model, X_test, y_test)

    elif mode == "📊 Compare Models":
        choices = st.multiselect("בחר מודלים:", ["RandomForest", "XGB", "LGBM", "CatBoost"], default=["RandomForest", "XGB", "LGBM", "CatBoost"])
        results = {}

        for choice in choices:
            if choice == "RandomForest":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(random_state=42)
            elif choice == "XGB":
                from xgboost import XGBClassifier
                model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
            elif choice == "LGBM":
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(random_state=42)
            elif choice == "CatBoost":
                from catboost import CatBoostClassifier
                model = CatBoostClassifier(verbose=0, random_state=42)

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            results[choice] = {"Accuracy": acc, "AUC": auc}

        results_df = pd.DataFrame(results).T
        st.dataframe(results_df)

        # ROC Comparison
        fig, ax = plt.subplots()
        for choice in choices:
            if choice == "RandomForest":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(random_state=42)
            elif choice == "XGB":
                from xgboost import XGBClassifier
                model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
            elif choice == "LGBM":
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(random_state=42)
            elif choice == "CatBoost":
                from catboost import CatBoostClassifier
                model = CatBoostClassifier(verbose=0, random_state=42)

            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{choice} (AUC={roc_auc:.2f})")

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_title("ROC Curves Comparison")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

# ============================
# 6. Streamlit UI
# ============================
st.title("🧠 Parkinson’s Prediction System")
st.markdown("מערכת מלאה: 🔍 EDA | 🔮 Prediction | 🎮 Playground")

uploaded_file = st.file_uploader("📂 העלה קובץ CSV/XLSX", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.subheader("📊 הצצה לנתונים")
    st.dataframe(data.head())

    tabEDA, tabPred, tabPlay = st.tabs(["🔍 EDA", "🔮 Prediction", "🎮 Playground"])

    with tabEDA:
        run_eda(data)

    with tabPred:
        if st.button("בצע חיזוי"):
            results, preds, probs = predict_with_risk(model, scaler, data)
            st.subheader("✅ תוצאות החיזוי")
            st.dataframe(results)

            # הורדות
            csv = results.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ הורד CSV", data=csv, file_name="predictions.csv", mime="text/csv")

            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                results.to_excel(writer, index=False)
            st.download_button("⬇️ הורד Excel", data=buffer.getvalue(), file_name="predictions.xlsx", mime="application/vnd.ms-excel")

            if "status" in data.columns:
                st.subheader("📈 Model Evaluation Curves")
                X_eval = scaler.transform(data.drop(columns=["status"]))
                y_eval = data["status"]
                plot_model_curves(model, X_eval, y_eval)

    with tabPlay:
        playground_ui(data)
