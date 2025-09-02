import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay
)

# ============================
# 1. טעינת מודל ו-Scaler
# ============================
try:
    model = joblib.load("models/best_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    st.success("✅ Model & Scaler loaded from /models/")
except Exception as e:
    st.error(f"❌ לא ניתן לטעון מודל/סקיילר: {e}")
    st.stop()

# ============================
# 2. Prediction עם Risk Labels
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
# 3. גרפים להערכת מודל
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
# 4. Streamlit UI
# ============================
st.title("🧠 Parkinson’s Prediction App (v16 Extension)")
st.markdown("חיזוי סטטוס מחלת פרקינסון עם 🟢🟡🔴 Risk Labels, טבלאות, גרפים והורדה ל-CSV/XLSX")

uploaded_file = st.file_uploader("📂 העלה קובץ CSV/XLSX עם נתוני מטופלים", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.subheader("📊 נתונים שהועלו")
    st.dataframe(data.head())

    if st.button("🔮 בצע חיזוי"):
        results, preds, probs = predict_with_risk(model, scaler, data)
        st.subheader("✅ תוצאות החיזוי")
        st.dataframe(results)

        # הורדה ל-CSV
        csv = results.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ הורד CSV", data=csv, file_name="predictions.csv", mime="text/csv")

        # הורדה ל-Excel
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            results.to_excel(writer, index=False)
        st.download_button("⬇️ הורד Excel", data=buffer.getvalue(), file_name="predictions.xlsx", mime="application/vnd.ms-excel")

        # אם יש "status" בנתונים – נציג עקומות
        if "status" in data.columns:
            st.subheader("📈 Model Evaluation Curves")
            X_eval = scaler.transform(data.drop(columns=["status"]))
            y_eval = data["status"]
            plot_model_curves(model, X_eval, y_eval)
        else:
            st.info("⚠️ לא נמצאה עמודת 'status' בקובץ – לא ניתן להציג עקומות ROC/PR/CM")
