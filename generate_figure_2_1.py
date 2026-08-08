# generate_figure_2_1.py
# Standalone script to generate Figure 2.1 (Class distribution before/after sampling)
# Author: Master Project - DDoS Detection

import pandas as pd
import matplotlib.pyplot as plt
import os

# === CONFIGURATION ===
BASE_DIR = r"D:\DDoS_Research_Project"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

print("🚀 Generating Figure 2.1 (Class Distribution)...")

# ============================================================================
# 1. LOAD DATA: DISTRIBUTION BEFORE SAMPLING
# ============================================================================
full_path = os.path.join(PROCESSED_DIR, "cicddos2019_full.parquet")
if os.path.exists(full_path):
    print("📂 Loading raw dataset (52.6 M rows)...")
    df_full = pd.read_parquet(full_path)
    n_normal_before = int((df_full['Label'] == 0).sum())
    n_attack_before = int((df_full['Label'] == 1).sum())
    total_before = n_normal_before + n_attack_before
    print(f"   ✅ Raw dataset: {total_before} rows")
else:
    # Fallback values (from execution logs)
    print("⚠️  Raw dataset not found, using stored values.")
    n_normal_before = 124883
    n_attack_before = 52503492
    total_before = n_normal_before + n_attack_before

pct_normal_before = (n_normal_before / total_before) * 100
pct_attack_before = (n_attack_before / total_before) * 100

print(f"   Before: Benign={n_normal_before} ({pct_normal_before:.4f}%)")
print(f"           Attacks={n_attack_before} ({pct_attack_before:.4f}%)")

# ============================================================================
# 2. LOAD DATA: DISTRIBUTION AFTER SAMPLING
# ============================================================================
y_train_path = os.path.join(FEATURES_DIR, "y_train.pkl")
if os.path.exists(y_train_path):
    y_train = pd.read_pickle(y_train_path)
    n_normal_after = int((y_train == 0).sum())
    n_attack_after = int((y_train == 1).sum())
    total_after = n_normal_after + n_attack_after
else:
    # Fallback values (from stratified sampling)
    print("⚠️  y_train.pkl not found, using stored values.")
    n_normal_after = 86503
    n_attack_after = 400000
    total_after = n_normal_after + n_attack_after

pct_normal_after = (n_normal_after / total_after) * 100
pct_attack_after = (n_attack_after / total_after) * 100

print(f"   After: Benign={n_normal_after} ({pct_normal_after:.2f}%)")
print(f"          Attacks={n_attack_after} ({pct_attack_after:.2f}%)")

# ============================================================================
# 3. GENERATE THE FIGURE
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
labels = ['Benign (Normal)', 'DDoS']
colors = ['#2E86C1', '#E74C3C']  # Blue for Benign, Red for DDoS
explode = (0.05, 0)  # Slightly explode the first slice

# --- Left: Before sampling ---
sizes_before = [pct_normal_before, pct_attack_before]
axes[0].pie(
    sizes_before,
    labels=labels,
    colors=colors,
    autopct='%1.4f%%',  # Precise display for small values
    startangle=90,
    explode=explode,
    shadow=True,
    textprops={'fontsize': 12, 'weight': 'bold'}
)
axes[0].set_title('Original Distribution (Raw)', fontsize=14, fontweight='bold')
axes[0].axis('equal')

# --- Right: After sampling ---
sizes_after = [pct_normal_after, pct_attack_after]
axes[1].pie(
    sizes_after,
    labels=labels,
    colors=colors,
    autopct='%1.2f%%',
    startangle=90,
    explode=explode,
    shadow=True,
    textprops={'fontsize': 12, 'weight': 'bold'}
)
axes[1].set_title('After Stratified Sampling (Training)', fontsize=14, fontweight='bold')
axes[1].axis('equal')

plt.suptitle('Figure 2.1 : Class balancing for training', fontsize=16, y=1.02)
plt.tight_layout()

# Save
output_path = os.path.join(FIGURES_DIR, 'fig1_class_distribution.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Figure saved to: {output_path}")
plt.close()

print("🎯 Done.")