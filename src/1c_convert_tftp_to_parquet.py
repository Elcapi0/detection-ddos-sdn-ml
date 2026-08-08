# 1c_convert_tftp_to_parquet.py
# Convertit TFTP.csv en un seul Parquet en écrivant d'abord des fichiers temporaires.

import pandas as pd
import numpy as np
import os
import glob

BASE_DIR = r"D:\DDoS_Research_Project"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TFTP_PATH = r"D:\DDoS_Research_Project\data\raw\01-12\TFTP.csv"
TEMP_DIR = os.path.join(PROCESSED_DIR, "tftp_temp")
os.makedirs(TEMP_DIR, exist_ok=True)

cols_to_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'Timestamp', 'SimillarHTTP']

print("🚀 Conversion de TFTP.csv en Parquet (streaming)...")
chunk_size = 100000
chunk_idx = 0

for i, chunk in enumerate(pd.read_csv(TFTP_PATH, chunksize=chunk_size, low_memory=False)):
    # Nettoyage
    chunk.columns = chunk.columns.str.strip()
    chunk = chunk.drop(columns=[c for c in cols_to_drop if c in chunk.columns])
    chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
    chunk['Label'] = chunk['Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
    
    # Écrire chaque chunk dans un fichier séparé
    part_path = os.path.join(TEMP_DIR, f"part_{i+1:04d}.parquet")
    chunk.to_parquet(part_path, engine='pyarrow', index=False)
    print(f"✅ Lot {i+1} : {len(chunk)} lignes écrites (Total : {(i+1)*chunk_size})")

# Fusionner tous les fichiers temporaires en un seul Parquet
print("🔄 Fusion des fichiers temporaires...")
all_files = glob.glob(os.path.join(TEMP_DIR, "*.parquet"))
df_list = [pd.read_parquet(f) for f in all_files]
df_tftp = pd.concat(df_list, ignore_index=True)

output_path = os.path.join(PROCESSED_DIR, "tftp_cleaned.parquet")
df_tftp.to_parquet(output_path, engine='pyarrow', index=False)
print(f"✅ TFTP final sauvegardé dans {output_path} avec {len(df_tftp)} lignes")

# Nettoyer les fichiers temporaires
import shutil
shutil.rmtree(TEMP_DIR)
print("🧹 Fichiers temporaires supprimés.")