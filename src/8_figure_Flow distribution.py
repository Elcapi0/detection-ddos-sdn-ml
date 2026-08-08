# Flow distribution across TASC cascade levels (Horizontal bar chart)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

# === CONFIGURATION ===
BASE_DIR = r"D:\DDoS_Research_Project"
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

print("🚀 Generating Figure 4.3 (Horizontal bar chart - TASC cascade)...")

# 1. Load data
print("📂 Loading data...")
X_val = pd.read_parquet(os.path.join(FEATURES_DIR, "X_val_50.parquet")).values
y_val = pd.read_pickle(os.path.join(FEATURES_DIR, "y_val.pkl")).values
print(f"   Validation: {len(X_val)} flows")

# 2. Load models
print("📂 Loading models...")
lgb = joblib.load(os.path.join(MODELS_DIR, "lgb_model.pkl"))
xgb = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
svm = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_svm.pkl"))

# 3. Run cascade to count flows per level
print("⏳ Running cascade...")
threshold_high = 0.95
threshold_low = 0.05

# Level 1: LightGBM
proba_lgb = lgb.predict_proba(X_val)[:, 1]
mask_stage1 = (proba_lgb > threshold_high) | (proba_lgb < threshold_low)

# Level 2: XGBoost (doubtful cases)
idx_doubtful = np.where(~mask_stage1)[0]
n_stage1 = np.sum(mask_stage1)
n_stage2 = 0
n_stage3 = 0

if len(idx_doubtful) > 0:
    proba_xgb = xgb.predict_proba(X_val[idx_doubtful])[:, 1]
    mask_xgb_high = proba_xgb > threshold_high
    mask_xgb_low = proba_xgb < threshold_low
    mask_stage2 = mask_xgb_high | mask_xgb_low
    n_stage2 = np.sum(mask_stage2)
    
    idx_stage3 = idx_doubtful[~mask_stage2]
    n_stage3 = len(idx_stage3)

total = n_stage1 + n_stage2 + n_stage3

print(f"\n📊 Flow distribution:")
print(f"   Level 1 (Early Exit) : {n_stage1} ({n_stage1/total*100:.4f}%)")
print(f"   Level 2 (XGBoost)    : {n_stage2} ({n_stage2/total*100:.4f}%)")
print(f"   Level 3 (SVM)        : {n_stage3} ({n_stage3/total*100:.4f}%)")

# 4. Generate horizontal bar chart
print("\n🖼️ Generating horizontal bar chart...")

# Data
levels = ['Level 1: LightGBM\n(Early Exit)', 'Level 2: XGBoost', 'Level 3: SVM']
values = [n_stage1, n_stage2, n_stage3]
percentages = [n_stage1/total*100, n_stage2/total*100, n_stage3/total*100]
colors = ['#2E86C1', '#F39C12', '#E74C3C']  # Blue, Orange, Red

# Create figure
fig, ax = plt.subplots(figsize=(10, 5))

# Horizontal bars
bars = ax.barh(levels, values, color=colors, edgecolor='black', linewidth=1.5, height=0.6)

# Add values and percentages on bars
for bar, val, pct in zip(bars, values, percentages):
    ax.text(
        bar.get_width() + (max(values) * 0.02),
        bar.get_y() + bar.get_height()/2,
        f'{val:,} flows  ({pct:.4f}%)',
        va='center',
        fontsize=12,
        weight='bold',
        color='black'
    )

# Add annotation on Level 1 bar to highlight Early Exit
ax.annotate(
    '⬅️ 99.96% of flows exit here',
    xy=(n_stage1 * 0.5, 0),
    xytext=(n_stage1 * 0.5, 0),
    ha='center',
    va='center',
    fontsize=13,
    weight='bold',
    color='white',
    bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.7)
)

# Customize axes
ax.set_xlabel('Number of Flows', fontsize=13)
ax.grid(axis='x', linestyle='--', alpha=0.6)

# X-axis limits to leave room for annotations
ax.set_xlim(0, max(values) * 1.35)

# Add summary box
metrics_text = (
    f"Total: {total:,} flows\n"
    f"Early Exit Rate: {percentages[0]:.2f}%\n"
    f"Flows handled by XGBoost: {percentages[1]:.4f}%\n"
    f"Flows handled by SVM: {percentages[2]:.4f}%"
)
props = dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9, edgecolor='gray')
ax.text(
    0.98, 0.02,
    metrics_text,
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment='bottom',
    horizontalalignment='right',
    bbox=props
)

plt.tight_layout()

# Save
output_path = os.path.join(FIGURES_DIR, 'fig4_3_bar_chart.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Figure saved to: {output_path}")
plt.close()

print("🎯 Done.")