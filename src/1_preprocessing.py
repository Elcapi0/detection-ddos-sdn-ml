# Objectif : Charger TOUS les fichiers CSV (01-12 + 03-11), nettoyer, fusionner et sauvegarder en .parquet

import pandas as pd
import numpy as np
import os
from glob import glob
import time

# === CONFIGURATION ===
BASE_DIR = r"D:\DDoS_Research_Project"
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Création du dossier de sortie s'il n'existe pas
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Colonnes à supprimer (identifiants et timestamps)
# Important : Les colonnes dans les CSVs ont parfois des espaces avant. On utilisera strip()
cols_to_drop = [
    'Flow ID', 'Source IP', 'Destination IP',
    'Source Port', 'Destination Port',
    'Timestamp', 'SimillarHTTP'
]

# === FONCTIONS DE NETTOYAGE ===
def clean_dataframe(df):
    """Nettoie un DataFrame : infinis, NaN, binarisation de la cible."""
    # 1. Remplacer les infinis par NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 2. Supprimer les lignes contenant des NaN (perte négligeable)
    df = df.dropna()
    
    # 3. Binarisation de la variable cible : 'BENIGN' -> 0, tout le reste (attaques) -> 1
    #    On utilise strip() pour éviter les problèmes d'espaces
    df['Label'] = df['Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
    
    return df

# === CHARGEMENT ET FUSION ===
def load_and_merge_all():
    print("🚀 Début du chargement des données...")
    start_total = time.time()
    
    df_full = None  # Variable qui contiendra le DataFrame final
    total_rows = 0
    list_folders = ['01-12', '03-11']
    
    for folder in list_folders:
        folder_path = os.path.join(RAW_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"⚠️  Dossier {folder_path} non trouvé, ignoré.")
            continue
            
        csv_files = glob(os.path.join(folder_path, "*.csv"))
        print(f"📂 Dossier {folder} : {len(csv_files)} fichiers trouvés.")
        
        for file_path in csv_files:
            print(f"   📄 Traitement de {os.path.basename(file_path)}...")
            start_file = time.time()
            
            try:
                # Lecture du CSV
                df = pd.read_csv(file_path, low_memory=False, encoding='utf-8')
                
                # --- Nettoyage des noms de colonnes ---
                # Suppression des espaces en début/fin de nom
                df.columns = df.columns.str.strip()
                
                # Suppression des colonnes identifiantes (si elles existent)
                cols_to_drop_existing = [col for col in cols_to_drop if col in df.columns]
                if cols_to_drop_existing:
                    df = df.drop(columns=cols_to_drop_existing)
                
                # Application du nettoyage (valeurs infinies, NaN, Label)
                df = clean_dataframe(df)
                
                # Fusion avec le DataFrame principal
                if df_full is None:
                    df_full = df
                else:
                    df_full = pd.concat([df_full, df], ignore_index=True)
                
                total_rows += len(df)
                print(f"      ✅ {len(df)} lignes ajoutées (Total : {total_rows}) - {time.time() - start_file:.2f}s")
                
            except Exception as e:
                print(f"      ❌ Erreur sur {file_path} : {e}")
    
    if df_full is None:
        print("❌ Aucune donnée chargée. Vérifiez les chemins.")
        return None

    # === SUPPRESSION DES DOUBLONS (optionnel mais recommandé) ===
    print("🧹 Suppression des doublons...")
    df_full = df_full.drop_duplicates()
    
    print("\n📊 Statistiques finales :")
    print(f"   - Nombre total d'échantillons : {len(df_full)}")
    print(f"   - Nombre de caractéristiques : {len(df_full.columns) - 1}")  # -1 pour la colonne Label
    print("   - Répartition des classes :")
    print(df_full['Label'].value_counts())
    
    # === SAUVEGARDE EN FORMAT PARQUET ===
    output_path = os.path.join(PROCESSED_DIR, "cicddos2019_full.parquet")
    print(f"💾 Sauvegarde du dataset dans {output_path}...")
    df_full.to_parquet(output_path, index=False)
    
    print(f"✅ Fusion terminée en {time.time() - start_total:.2f} secondes.")
    print(f"   Fichier sauvegardé : {output_path}")
    return df_full

# --- EXÉCUTION ---
if __name__ == "__main__":
    # Petit check pour être sûr que les chemins existent
    if not os.path.exists(RAW_DIR):
        print(f"❌ ERREUR : Le dossier {RAW_DIR} n'existe pas.")
        print("   Veuillez vérifier que vos dossiers 01-12 et 03-11 sont dans data/raw/")
    else:
        load_and_merge_all()