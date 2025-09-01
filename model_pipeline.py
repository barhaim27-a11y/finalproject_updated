# ============================
# 📦 model_pipeline.py (v17)
# ============================
import os, json, joblib, shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, classification_report
)

# מודלים
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

BASE_DIR = "."
DATA_PATH = os.path.join(BASE_DIR, "data", "parkinsons.csv")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================
# 1. Load dataset
# ============================
df = pd.read_csv(DATA_PATH)
X = df.drop("status", axis=1)
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# 2. Define models
# ============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVC": SVC(probability=True, kernel="rbf"),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
}

# ============================
# 3. Train & evaluate
# ============================
leaderboard = {}
results = {}

for name, model in models.items():
    print(f"Training {name}...")
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    pipe.fit(X_train, y_train)

    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)
    y_prob_train = pipe.predict_proba(X_train)[:,1]
    y_prob_test = pipe.predict_proba(X_test)[:,1]

    metrics_train = {
        "accuracy": accuracy_score(y_train, y_pred_train),
        "precision": precision_score(y_train, y_pred_train),
        "recall": recall_score(y_train, y_pred_train),
        "f1": f1_score(y_train, y_pred_train),
        "roc_auc": roc_auc_score(y_train, y_prob_train),
    }
    metrics_test = {
        "accuracy": accuracy_score(y_test, y_pred_test),
        "precision": precision_score(y_test, y_pred_test),
        "recall": recall_score(y_test, y_pred_test),
        "f1": f1_score(y_test, y_pred_test),
        "roc_auc": roc_auc_score(y_test, y_prob_test),
    }

    leaderboard[name] = {"train": metrics_train, "test": metrics_test}
    results[name] = metrics_test["roc_auc"]

# ============================
# 4. Save leaderboard & best model
# ============================
best_name = max(results, key=results.get)
best_model = Pipeline([("scaler", StandardScaler()), ("clf", models[best_name])])
best_model.fit(X_train, y_train)

joblib.dump(best_model, os.path.join(MODELS_DIR,"best_model.joblib"))

with open(os.path.join(ASSETS_DIR,"metrics.json"), "w") as f:
    json.dump(leaderboard[best_name], f, indent=2)

with open(os.path.join(ASSETS_DIR,"leaderboard.json"), "w") as f:
    json.dump(leaderboard, f, indent=2)

# ============================
# 5. Plots
# ============================
# ROC
y_prob = best_model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test,y_prob):.2f}")
plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(ASSETS_DIR,"roc_curve.png"))
plt.close()

# PR Curve
prec, rec, _ = precision_recall_curve(y_test, y_prob)
plt.figure()
plt.plot(rec, prec)
plt.title("Precision-Recall Curve")
plt.savefig(os.path.join(ASSETS_DIR,"pr_curve.png"))
plt.close()

# Confusion Matrix
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy","Parkinson’s"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(ASSETS_DIR,"confusion_matrix.png"))
plt.close()

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(best_model, X, y, cv=3, scoring="roc_auc")
plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1), label="Train")
plt.plot(train_sizes, test_scores.mean(axis=1), label="Test")
plt.title("Learning Curve")
plt.legend()
plt.savefig(os.path.join(ASSETS_DIR,"learning_curve.png"))
plt.close()

# ============================
# 6. SHAP Explainability (fallback)
# ============================
model_to_explain = best_model.named_steps["clf"]
try:
    if isinstance(model_to_explain, (RandomForestClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier, LogisticRegression)):
        explainer = shap.TreeExplainer(model_to_explain)
        shap_values = explainer(X_test)
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig(os.path.join(ASSETS_DIR,"shap_summary.png"), bbox_inches="tight")
        plt.close()
    else:
        explainer = shap.KernelExplainer(model_to_explain.predict_proba, shap.sample(X_train, 50))
        shap_values = explainer.shap_values(X_test[:20])
        shap.summary_plot(shap_values, X_test[:20], show=False)
        plt.savefig(os.path.join(ASSETS_DIR,"shap_summary.png"), bbox_inches="tight")
        plt.close()
except Exception as e:
    print(f"SHAP failed: {e}")
