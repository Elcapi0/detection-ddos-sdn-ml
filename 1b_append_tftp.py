# 1b_append_tftp.py
# Charge le fichier TFTP.csv par morceaux pour éviter l'erreur mémoire

import pandas as pd
import numpy as np
import os

BASE_DIR = r"D:\DDoS_Research_Project"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TFTP_PATH = r"D:\DDoS_Research_Project\data\raw\01-12\TFTP.csv"

# Chargement du Parquet existant
df_full = pd.read_parquet(os.path.join(PROCESSED_DIR, "cicddos2019_full.parquet"))
print(f"Base actuelle : {len(df_full)} lignes")

# Chargement de TFTP en chunks (50 000 lignes par lot)
chunk_size = 50000
total_appended = 0

print("Chargement de TFTP.csv par morceaux...")
for i, chunk in enumerate(pd.read_csv(TFTP_PATH, chunksize=chunk_size, low_memory=False)):
    # Nettoyage des colonnes
    chunk.columns = chunk.columns.str.strip()
    cols_to_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'Timestamp', 'SimillarHTTP']
    chunk = chunk.drop(columns=[c for c in cols_to_drop if c in chunk.columns])
    
    # Nettoyage des valeurs
    chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Binarisation
    chunk['Label'] = chunk['Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
    
    # Concaténation
    df_full = pd.concat([df_full, chunk], ignore_index=True)
    total_appended += len(chunk)
    print(f"  Lot {i+1} : {len(chunk)} lignes ajoutées (Total TFTP : {total_appended})")

# Sauvegarde finale
df_full.drop_duplicates(inplace=True)
df_full.to_parquet(os.path.join(PROCESSED_DIR, "cicddos2019_full.parquet"), index=False)
print(f"✅ Fusion terminée. Nouvelle taille : {len(df_full)} lignes")