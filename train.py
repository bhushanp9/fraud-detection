"""
train.py — Credit Card Fraud Detection
---------------------------------------
Trains a RandomForestClassifier on the Kaggle Credit Card Fraud dataset.
Saves the trained model as fraud_detection_model.pkl

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATA_PATH  = os.getenv("DATA_PATH",  "creditcard.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "fraud_detection_model.pkl")
TEST_SIZE  = 0.2
RANDOM_STATE = 42

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   Rows: {len(df):,} | Columns: {df.shape[1]}")
print(f"   Fraud cases : {df['Class'].sum():,}  ({df['Class'].mean()*100:.3f}%)")
print(f"   Legit cases : {(df['Class'] == 0).sum():,}")

# ─────────────────────────────────────────
# 2. FEATURES & TARGET
# ─────────────────────────────────────────
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET   = "Class"

X = df[FEATURES]
y = df[TARGET]

# ─────────────────────────────────────────
# 3. SCALE Time & Amount
#    (V1–V28 are already PCA-transformed)
# ─────────────────────────────────────────
print("\n⚙️  Scaling Time and Amount features...")
scaler = StandardScaler()
X = X.copy()
X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

# ─────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
#    stratify=y keeps fraud ratio in both splits
# ─────────────────────────────────────────
print("✂️  Splitting into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)
print(f"   Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

# ─────────────────────────────────────────
# 5. TRAIN MODEL
#    class_weight='balanced' handles the
#    heavy class imbalance (fraud is ~0.17%)
# ─────────────────────────────────────────
print("\n🌲 Training RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",   # handles imbalance without SMOTE
    random_state=RANDOM_STATE,
    n_jobs=-1,                 # use all CPU cores
)
model.fit(X_train, y_train)
print("   Training complete ✅")

# ─────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────
print("\n📊 Evaluation on test set:")
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraudulent"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"  True Negatives  (correctly flagged legit)  : {cm[0][0]:,}")
print(f"  False Positives (legit flagged as fraud)   : {cm[0][1]:,}")
print(f"  False Negatives (fraud missed)             : {cm[1][0]:,}")
print(f"  True Positives  (fraud correctly caught)   : {cm[1][1]:,}")

roc = roc_auc_score(y_test, y_proba)
print(f"\nROC-AUC Score : {roc:.4f}")

# Top 5 most important features
feature_importance = pd.Series(model.feature_importances_, index=FEATURES)
top5 = feature_importance.nlargest(5)
print("\nTop 5 Important Features:")
for feat, score in top5.items():
    print(f"  {feat:<10} {score:.4f}")

# ─────────────────────────────────────────
# 7. SAVE MODEL
# ─────────────────────────────────────────
print(f"\n💾 Saving model to {MODEL_PATH}...")
joblib.dump(model, MODEL_PATH)
print(f"   Model saved ✅  ({os.path.getsize(MODEL_PATH) / 1e6:.1f} MB)")
print("\n🎉 Done! Run app.py to start the API.")
