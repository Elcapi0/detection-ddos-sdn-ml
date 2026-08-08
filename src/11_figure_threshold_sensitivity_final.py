# Script ULTIME pour la Figure 4.4 (Sensibilité au seuil)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.metrics import accuracy_score, f1_score

# === CONFIGURATION ===
BASE_DIR = r"D:\DDoS_Research_Project"
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

print("🚀 Génération de la Figure 4.4 (Sensibilité au seuil)...")

# 1. Chargement des données
print("📂 Chargement des données...")
X_val = pd.read_parquet(os.path.join(FEATURES_DIR, "X_val_50.parquet")).values
y_val = pd.read_pickle(os.path.join(FEATURES_DIR, "y_val.pkl")).values
print(f"   Validation : {len(X_val)} lignes")

# 2. Chargement des modèles
print("📂 Chargement des modèles...")
lgb = joblib.load(os.path.join(MODELS_DIR, "lgb_model.pkl"))
xgb = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))

# 3. Prédictions de base
proba_lgb = lgb.predict_proba(X_val)[:, 1]

# 4. Étude de sensibilité
thresholds = np.arange(0.50, 0.99, 0.02)
early_exit_rates = []
accuracies = []
f1_scores = []

for th in thresholds:
    mask_stage1 = (proba_lgb > th) | (proba_lgb < (1 - th))
    early_exit_rates.append(np.sum(mask_stage1) / len(X_val))
    
    preds = np.zeros(len(X_val), dtype=int)
    preds[proba_lgb > th] = 1
    preds[proba_lgb < (1 - th)] = 0
    
    idx_doubtful = np.where(~mask_stage1)[0]
    if len(idx_doubtful) > 0:
        proba_xgb = xgb.predict_proba(X_val[idx_doubtful])[:, 1]
        for j, idx in enumerate(idx_doubtful):
            preds[idx] = 1 if proba_xgb[j] > 0.5 else 0
    
    accuracies.append(accuracy_score(y_val, preds))
    f1_scores.append(f1_score(y_val, preds))

# 5. Génération de la figure
fig, ax1 = plt.subplots(figsize=(11, 7))

bars = ax1.bar(thresholds, early_exit_rates, width=0.02, alpha=0.4, color='#2E86C1', label='Early Exit Rate')
ax1.set_xlabel('Confidence Threshold', fontsize=13)
ax1.set_ylabel('Early Exit Rate (%)', color='#2E86C1', fontsize=13)
ax1.tick_params(axis='y', labelcolor='#2E86C1')
ax1.set_ylim(0.95, 1.01)

ax2 = ax1.twinx()
ax2.plot(thresholds, accuracies, 'o-', color='#E74C3C', linewidth=2.5, markersize=8, label='Accuracy')
ax2.plot(thresholds, f1_scores, 's-', color='#27AE60', linewidth=2.5, markersize=8, label='F1-Score')
ax2.set_ylabel('Performance (Accuracy / F1-Score)', color='black', fontsize=13)
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_ylim(0.9980, 1.0005)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.2f}%'))

ax1.axvline(x=0.95, color='black', linestyle='--', linewidth=2, label='Optimal threshold (0.95)')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=11)

plt.title('Figure 4.4 : Impact of Early Exit threshold on performance and filtering rate', fontsize=15, pad=20)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

output_path = os.path.join(FIGURES_DIR, 'fig4_4_threshold_sensitivity_final.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Figure sauvegardée dans : {output_path}")
plt.close()