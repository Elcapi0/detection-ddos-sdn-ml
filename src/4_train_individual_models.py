# Entraîne LightGBM, XGBoost et SVM séparément pour la cascade

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import os
import time

BASE_DIR = r"D:\DDoS_Research_Project"
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("="*60)
print("🚀 ENTRAÎNEMENT DES MODÈLES INDIVIDUELS")
print("="*60)

# Chargement
X_train = pd.read_parquet(os.path.join(FEATURES_DIR, "X_train_50.parquet")).values
y_train = pd.read_pickle(os.path.join(FEATURES_DIR, "y_train.pkl")).values
X_val = pd.read_parquet(os.path.join(FEATURES_DIR, "X_val_50.parquet")).values
y_val = pd.read_pickle(os.path.join(FEATURES_DIR, "y_val.pkl")).values

print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# 1. LIGHTGBM
print("\n⏳ Entraînement de LightGBM...")
lgb = LGBMClassifier(
    n_estimators=200, num_leaves=31, learning_rate=0.05,
    random_state=42, verbose=-1, n_jobs=-1
)
lgb.fit(X_train, y_train)
joblib.dump(lgb, os.path.join(MODELS_DIR, "lgb_model.pkl"))
print("✅ LightGBM sauvegardé")

# 2. XGBOOST
print("\n⏳ Entraînement de XGBoost...")
xgb = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    random_state=42, eval_metric='logloss', n_jobs=-1
)
xgb.fit(X_train, y_train)
joblib.dump(xgb, os.path.join(MODELS_DIR, "xgb_model.pkl"))
print("✅ XGBoost sauvegardé")

# 3. SVM (nécessite normalisation)
print("\n⏳ Normalisation pour SVM...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("⏳ Entraînement de SVM (peut prendre 5-10 min)...")
svm = SVC(
    kernel='rbf', C=10, gamma=0.1,
    probability=True, random_state=42
)
svm.fit(X_train_scaled, y_train)
joblib.dump(svm, os.path.join(MODELS_DIR, "svm_model.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_svm.pkl"))
print("✅ SVM sauvegardé")

print("\n" + "="*60)
print("🎯 MODÈLES INDIVIDUELS PRÊTS")
print("="*60)