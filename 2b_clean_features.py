# 2b_clean_features.py
# Supprime 'Unnamed: 0' et les colonnes constantes des datasets déjà sauvegardés.

import pandas as pd
import os
import joblib

FEATURES_DIR = r"D:\DDoS_Research_Project\data\features"

# Charger les données
X_train = pd.read_parquet(os.path.join(FEATURES_DIR, "X_train_50.parquet"))
y_train = pd.read_pickle(os.path.join(FEATURES_DIR, "y_train.pkl"))
X_val = pd.read_parquet(os.path.join(FEATURES_DIR, "X_val_50.parquet"))
y_val = pd.read_pickle(os.path.join(FEATURES_DIR, "y_val.pkl"))

print(f"Taille avant nettoyage : {X_train.shape}")

# 1. Supprimer 'Unnamed: 0' si elle existe
if 'Unnamed: 0' in X_train.columns:
    X_train = X_train.drop(columns=['Unnamed: 0'])
    X_val = X_val.drop(columns=['Unnamed: 0'])
    print("✅ Suppression de 'Unnamed: 0'")

# 2. Supprimer les colonnes constantes (variance = 0)
# On les identifie sur X_train
constant_cols = [col for col in X_train.columns if X_train[col].std() == 0]
if constant_cols:
    X_train = X_train.drop(columns=constant_cols)
    X_val = X_val.drop(columns=constant_cols)
    print(f"✅ Suppression des colonnes constantes : {constant_cols}")

print(f"Taille après nettoyage : {X_train.shape}")

# Sauvegarde (écrase les anciens fichiers)
X_train.to_parquet(os.path.join(FEATURES_DIR, "X_train_50.parquet"))
X_val.to_parquet(os.path.join(FEATURES_DIR, "X_val_50.parquet"))
pd.Series(y_train).to_pickle(os.path.join(FEATURES_DIR, "y_train.pkl"))
pd.Series(y_val).to_pickle(os.path.join(FEATURES_DIR, "y_val.pkl"))

print("💾 Données corrigées sauvegardées.")