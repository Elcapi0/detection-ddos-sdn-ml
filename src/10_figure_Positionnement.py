# Script standalone pour générer la Figure 4.6 (Positionnement par rapport à l'état de l'art)

import matplotlib.pyplot as plt
import numpy as np
import os

# === CONFIGURATION ===
BASE_DIR = r"D:\DDoS_Research_Project"
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

print("🚀 Génération de la Figure 4.6 (Positionnement par rapport à l'état de l'art)...")

# 1. Données de comparaison (Accuracy en pourcentage)
studies = [
    "Our study\n(TASC)",
    "Riziq & Ichsan\n(2025)",
    "En-CNN\n(2025)",
    "Alomari et al.\n(2025)",
    "Kaur et al.\n(2021)",
    "Sharma et al.\n(2022)"
]

accuracies = [99.98, 99.96, 99.98, 99.67, 99.80, 99.30]

# Couleurs : mettre en avant votre contribution (barre rouge)
colors = ['#E74C3C', '#2E86C1', '#2E86C1', '#2E86C1', '#2E86C1', '#2E86C1']

# 2. Génération du graphique à barres
fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.bar(studies, accuracies, color=colors, edgecolor='black', linewidth=1.5, width=0.6)

# Ajout des valeurs sur les barres
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.08,
            f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, weight='bold')

# Configuration des axes
ax.set_ylim(98.5, 100.5)
ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Ajout d'une ligne de référence pour l'état de l'art
ax.axhline(y=99.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Reference threshold (99.5%)')
ax.legend(loc='lower right', fontsize=10)

plt.tight_layout()

# Sauvegarde
output_path = os.path.join(FIGURES_DIR, 'plfig4_6_state_of_art_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Figure sauvegardée dans : {output_path}")
plt.close()

print("🎯 Terminé.")