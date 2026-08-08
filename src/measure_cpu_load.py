# measure_cpu_load.py
# Mesure précise du temps CPU pour prouver la réduction de charge CPU
# Comparaison : TASC (Early Exit) vs Stacking (tous les modèles sur tous les flux)

import pandas as pd
import numpy as np
import time
import os
import joblib
import gc

# === CONFIGURATION ===
BASE_DIR = r"D:\DDoS_Research_Project"
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")

print("="*70)
print("🚀 MESURE DU TEMPS CPU - TASC vs STACKING")
print("="*70)

# 1. Chargement des données
print("\n📂 Chargement des données...")
X_val = pd.read_parquet(os.path.join(FEATURES_DIR, "X_val_50.parquet")).values
y_val = pd.read_pickle(os.path.join(FEATURES_DIR, "y_val.pkl")).values
print(f"   Validation : {len(X_val)} lignes")

# 2. Chargement des modèles
print("📂 Chargement des modèles...")
lgb = joblib.load(os.path.join(MODELS_DIR, "lgb_model.pkl"))
xgb = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
svm = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_svm.pkl"))

print("\n" + "="*70)
print("📊 MESURE DU TEMPS CPU (process_time)")
print("="*70)

# === 3. Fonction TASC (Early Exit) ===
def tasc_predict():
    # Niveau 1 : LightGBM sur TOUS les flux
    proba_lgb = lgb.predict_proba(X_val)[:, 1]
    
    # Early Exit : flux sûrs
    mask_stage1 = (proba_lgb > 0.95) | (proba_lgb < 0.05)
    idx_doubtful = np.where(~mask_stage1)[0]
    
    # Niveau 2 : XGBoost sur les cas douteux
    if len(idx_doubtful) > 0:
        proba_xgb = xgb.predict_proba(X_val[idx_doubtful])[:, 1]
        mask_xgb_high = proba_xgb > 0.95
        mask_xgb_low = proba_xgb < 0.05
        mask_xgb_resolved = mask_xgb_high | mask_xgb_low
        idx_stage3 = idx_doubtful[~mask_xgb_resolved]
        
        # Niveau 3 : SVM sur les cas très rares
        if len(idx_stage3) > 0:
            X_scaled = scaler.transform(X_val[idx_stage3])
            _ = svm.predict_proba(X_scaled)
    
    return len(X_val)  # Retourne le nombre de flux pour vérification

# === 4. Fonction Stacking (tous les modèles sur TOUS les flux) ===
def stacking_predict():
    # LightGBM sur TOUS les flux
    _ = lgb.predict_proba(X_val)
    # XGBoost sur TOUS les flux
    _ = xgb.predict_proba(X_val)
    # SVM sur TOUS les flux (nécessite normalisation)
    X_scaled = scaler.transform(X_val)
    _ = svm.predict_proba(X_scaled)
    return len(X_val)

# === 5. Exécution des tests (5 itérations chacun) ===
print("\n⏳ Exécution des tests...")

# TASC
tasc_times = []
for i in range(5):
    gc.collect()  # Nettoyer la mémoire
    start = time.process_time()
    n_flows = tasc_predict()
    end = time.process_time()
    tasc_times.append(end - start)
    print(f"   TASC itération {i+1} : {end - start:.4f} s")

# Stacking
stack_times = []
for i in range(5):
    gc.collect()  # Nettoyer la mémoire
    start = time.process_time()
    n_flows = stacking_predict()
    end = time.process_time()
    stack_times.append(end - start)
    print(f"   Stacking itération {i+1} : {end - start:.4f} s")

# === 6. Résultats ===
tasc_mean = np.mean(tasc_times)
tasc_std = np.std(tasc_times)
stack_mean = np.mean(stack_times)
stack_std = np.std(stack_times)
ratio = stack_mean / tasc_mean
reduction = (1 - 1/ratio) * 100

print("\n" + "="*70)
print("📊 RÉSULTATS COMPARATIFS")
print("="*70)
print(f"\n   TASC    : {tasc_mean:.4f} s (±{tasc_std:.4f})")
print(f"   Stacking: {stack_mean:.4f} s (±{stack_std:.4f})")
print(f"\n   🚀 TASC est {ratio:.1f}x plus rapide que Stacking !")
print(f"   🎯 Réduction de charge CPU : {reduction:.1f}%")

print("\n" + "="*70)
print("🎯 PREUVE DE LA RÉDUCTION DE CHARGE CPU")
print("="*70)