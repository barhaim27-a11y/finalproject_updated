# ============================
# 📦 model_pipeline.py – Final v15
# ============================

import os, json, joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ============================
# Paths
# ============================
BASE_DIR = "."
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================
# Load Dataset
# ============================
data_path = os.path.join(DATA_DIR, "parkinsons.csv")
if not os.path.exists(data_path):
    raise FileNotFoundError("⚠️ Dataset not found. Please make sure 'data/parkinsons.csv' exists.")

df = pd.read_csv(data_path)
if "name" in df.columns:
    df = df.drop(columns=["name"])

X = df.drop("status", axis=1)
y = df["status"]

# ============================
# Models
# ============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVC": SVC(probability=True, kernel='rbf'),
    "MLP (Sklearn)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
}

# ============================
# Training & Metrics
# ============================
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    y_prob = pipe.predict_proba(X)[:, 1]

    results[name] = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_prob)
    }

# Save metrics & leaderboard
with open(os.path.join(ASSETS_DIR, "metrics.json"), "w") as f:
    json.dump(results, f, indent=4)

with open(os.path.join(ASSETS_DIR, "leaderboard.json"), "w") as f:
    json.dump(results, f, indent=4)

# ============================
# Best Model
# ============================
best_name = max(results, key=lambda k: results[k]["roc_auc"])
print(f"🏆 Best model: {best_name}")

best_model = Pipeline([("scaler", StandardScaler()), ("clf", models[best_name])])
best_model.fit(X, y)
joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))

print("✅ Training complete. Metrics & model saved.")
