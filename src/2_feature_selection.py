# 2_feature_selection.py
# Stratégie : Échantillonner les attaques DANS chaque fichier avant de les fusionner.
# Cela évite de charger les 52 millions de lignes en RAM.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os
import time

BASE_DIR = r"D:\DDoS_Research_Project"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
os.makedirs(FEATURES_DIR, exist_ok=True)

print("🚀 Chargement et échantillonnage intelligent...")
start_total = time.time()

# === 1. CHARGEMENT DE LA BASE (33M) ===
print("📂 Chargement de la Base (33M lignes)...")
df_base = pd.read_parquet(os.path.join(PROCESSED_DIR, "cicddos2019_full.parquet"))
print(f"   Base chargée : {len(df_base)} lignes.")

print("🎯 Échantillonnage des attaques dans la Base...")
df_normal_base = df_base[df_base['Label'] == 0]  # ~83 000 lignes
df_attack_base = df_base[df_base['Label'] == 1]  # ~33 000 000 lignes

# On prend 400 000 attaques aléatoires dans la Base (pour la diversité des attaques DNS, SYN, etc.)
df_attack_sample_base = df_attack_base.sample(n=400000, random_state=42)
del df_base, df_attack_base  # Libération immédiate de la RAM
print(f"   Base échantillonnée : {len(df_normal_base)} Normaux, {len(df_attack_sample_base)} Attaques")

# === 2. CHARGEMENT DE TFTP (19.5M) ===
print("📂 Chargement de TFTP (19.5M lignes)...")
df_tftp = pd.read_parquet(os.path.join(PROCESSED_DIR, "tftp_cleaned.parquet"))
print(f"   TFTP chargé : {len(df_tftp)} lignes.")

print("🎯 Échantillonnage des attaques dans TFTP...")
df_attack_tftp = df_tftp[df_tftp['Label'] == 1]  # TFTP ne contient quasi que des attaques
df_normal_tftp = df_tftp[df_tftp['Label'] == 0]  # (très probablement vide)

# On prend 100 000 attaques aléatoires dans TFTP (pour couvrir ce type d'attaque)
df_attack_sample_tftp = df_attack_tftp.sample(n=100000, random_state=42)
del df_tftp, df_attack_tftp
print(f"   TFTP échantillonné : {len(df_normal_tftp)} Normaux, {len(df_attack_sample_tftp)} Attaques")

# === 3. FUSION DES ÉCHANTILLONS (Taille finale : ~583 000 lignes) ===
print("🔄 Fusion des échantillons...")
df_normal = pd.concat([df_normal_base, df_normal_tftp], ignore_index=True)
df_attack = pd.concat([df_attack_sample_base, df_attack_sample_tftp], ignore_index=True)

# Assemblage final
df_train = pd.concat([df_normal, df_attack]).sample(frac=1, random_state=42)
print(f"✅ Dataset d'entraînement final : {len(df_train)} lignes")
print(f"   Répartition : Normaux = {len(df_normal)}, Attaques = {len(df_attack)}")

# Libération des DataFrames intermédiaires
del df_normal, df_attack, df_normal_base, df_normal_tftp, df_attack_sample_base, df_attack_sample_tftp

# === 4. SÉPARATION X / y ===
X = df_train.drop(columns=['Label'])
y = df_train['Label']
del df_train

# === 5. SPLIT TRAIN / VALIDATION (80/20) ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === 6. SÉLECTION DES TOP 50 FEATURES ===
print("🔍 Sélection des 50 meilleures caractéristiques (SelectKBest)...")
selector = SelectKBest(f_classif, k=50)
X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)

selected_features = X_train.columns[selector.get_support()].tolist()
print(f"✅ {len(selected_features)} features sélectionnées :")
print(selected_features)

# === 7. SAUVEGARDE ===
pd.DataFrame(X_train_selected, columns=selected_features).to_parquet(
    os.path.join(FEATURES_DIR, "X_train_50.parquet"))
pd.DataFrame(X_val_selected, columns=selected_features).to_parquet(
    os.path.join(FEATURES_DIR, "X_val_50.parquet"))
pd.Series(y_train).to_pickle(os.path.join(FEATURES_DIR, "y_train.pkl"))
pd.Series(y_val).to_pickle(os.path.join(FEATURES_DIR, "y_val.pkl"))

joblib.dump(selector, os.path.join(FEATURES_DIR, "selector_50.pkl"))
print(f"💾 Données sauvegardées dans {FEATURES_DIR}")
print(f"✅ Terminé en {time.time() - start_total:.2f} secondes.")