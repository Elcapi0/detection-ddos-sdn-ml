# Script standalone pour générer la Figure 4.5 (Compromis Accuracy / Efficacité computationnelle)

import matplotlib.pyplot as plt
import numpy as np
import os

# === CONFIGURATION ===
BASE_DIR = r"D:\DDoS_Research_Project"
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

print("🚀 Génération de la Figure 4.5 (Compromis Accuracy / Efficacité computationnelle)...")

# 1. Données mesurées
models = ['XGBoost', 'LightGBM', 'Stacking', 'TASC (Cascade)']
accuracies = [0.9993, 0.9992, 0.9999, 0.9998]
times_us = [0.43, 0.82, 1.23, 1.22]

# 2. Création de la figure
fig, ax1 = plt.subplots(figsize=(10, 6))

# Barres pour la latence (axe gauche)
bar_width = 0.6
colors = ['#3498DB', '#2ECC71', '#F39C12', '#E74C3C']
bars = ax1.bar(models, times_us, width=bar_width, color=colors, alpha=0.85, label='Latence (µs/flux)')
ax1.set_xlabel('Modèles', fontsize=13)
ax1.set_ylabel('Average Latency (microseconds / flow)', color='black', fontsize=13)
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_ylim(0, max(times_us) * 1.5)

# Valeurs de latence à l'intérieur des barres (en blanc pour lisibilité)
for bar, val in zip(bars, times_us):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() / 2,
             f'{val:.2f} µs', ha='center', va='center', fontsize=10, 
             weight='bold', color='white')

# Deuxième axe pour l'Accuracy
ax2 = ax1.twinx()
ax2.plot(models, accuracies, 'o-', color='purple', linewidth=3, markersize=12, label='Accuracy')
ax2.set_ylabel('Accuracy', color='purple', fontsize=13)
ax2.tick_params(axis='y', labelcolor='purple')
ax2.set_ylim(0.9980, 1.0005)

# Formatage de l'axe Y en pourcentage
from matplotlib.ticker import FuncFormatter
def perc_formatter(x, pos):
    return f'{x*100:.2f}%'
ax2.yaxis.set_major_formatter(FuncFormatter(perc_formatter))

# Valeurs d'accuracy AU-DESSUS des points (avec décalage vertical)
offset = 0.00045
for i, (model, acc) in enumerate(zip(models, accuracies)):
    ax2.text(i, acc + offset, f'{acc*100:.2f}%', ha='center', va='bottom', 
             fontsize=10, color='purple', weight='bold')

# Ajout de la mention "Early Exit: 99.96%" pour le TASC
ax1.text(3, times_us[3] + 0.25, 'Early Exit: 99.96%', ha='center', va='bottom',
         fontsize=9, color='#E74C3C', weight='bold', style='italic')

# Titre

# Légendes combinées
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
           bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=11)

plt.tight_layout()

# Sauvegarde
output_path = os.path.join(FIGURES_DIR, 'fig4_5_accuracy_vs_time.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Figure sauvegardée dans : {output_path}")
plt.close()

print("🎯 Terminé.")