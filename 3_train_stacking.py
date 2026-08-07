# 3_train_stacking.py
# Objectif : Entraîner un Stacking Classifier (LGB + XGB + RF) pour atteindre 99%+ sur CICDDoS2019.
# Auteur : Projet Master DDoS Detection

import pandas as pd
import numpy as np
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os
import time

# === CONFIGURATION ===
BASE_DIR = r"D:\DDoS_Research_Project"
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("="*60)
print("🚀 LANCEMENT DE L'ENTRAÎNEMENT DU STACKING")
print("="*60)

# 1. CHARGEMENT DES DONNÉES
print("\n📂 Chargement des données...")
X_train = pd.read_parquet(os.path.join(FEATURES_DIR, "X_train_50.parquet")).values
y_train = pd.read_pickle(os.path.join(FEATURES_DIR, "y_train.pkl")).values
X_val = pd.read_parquet(os.path.join(FEATURES_DIR, "X_val_50.parquet")).values
y_val = pd.read_pickle(os.path.join(FEATURES_DIR, "y_val.pkl")).values

print(f"   Train : {X_train.shape[0]} lignes, {X_train.shape[1]} features")
print(f"   Validation : {X_val.shape[0]} lignes")
print(f"   Répartition Train : Normaux={np.sum(y_train==0)}, Attaques={np.sum(y_train==1)}")

# 2. DÉFINITION DU STACKING
print("\n🔧 Construction du Stacking Classifier...")
base_learners = [
    ('lgb', LGBMClassifier(
        n_estimators=200, num_leaves=31, learning_rate=0.05,
        random_state=42, verbose=-1, n_jobs=-1
    )),
    ('xgb', XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, use_label_encoder=False, eval_metric='logloss',
        n_jobs=-1
    )),
    ('rf', RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
    ))
]

meta_learner = LogisticRegression(max_iter=1000, random_state=42)

stacking_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,  # Validation croisée à 5 plis pour le méta-modèle
    stack_method='predict_proba',
    n_jobs=-1
)

# 3. ENTRAÎNEMENT
print("\n⏳ Entraînement du Stacking (20-30 min selon votre machine)...")
start_time = time.time()
stacking_model.fit(X_train, y_train)
duration = time.time() - start_time
print(f"✅ Entraînement terminé en {duration:.2f} secondes ({duration/60:.2f} min)")

# 4. ÉVALUATION SUR VALIDATION
print("\n📊 ÉVALUATION SUR L'ENSEMBLE DE VALIDATION")
y_pred = stacking_model.predict(X_val)
y_proba = stacking_model.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred)
rec = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_proba)

print(f"   Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
print(f"   Precision : {prec:.4f}")
print(f"   Recall    : {rec:.4f}")
print(f"   F1-Score  : {f1:.4f}")
print(f"   ROC-AUC   : {auc:.4f}")

# 5. MATRICE DE CONFUSION
cm = confusion_matrix(y_val, y_pred)
tn, fp, fn, tp = cm.ravel()
print("\n📋 Matrice de confusion :")
print(f"   Vrais Positifs  (TP) : {tp}")
print(f"   Vrais Négatifs  (TN) : {tn}")
print(f"   Faux Positifs   (FP) : {fp}")
print(f"   Faux Négatifs   (FN) : {fn}")

# 6. SAUVEGARDE
model_path = os.path.join(MODELS_DIR, "stacking_model.pkl")
joblib.dump(stacking_model, model_path)
print(f"\n💾 Modèle sauvegardé dans : {model_path}")

print("\n" + "="*60)
print("🎯 ENTRAÎNEMENT TERMINÉ")
print("="*60)