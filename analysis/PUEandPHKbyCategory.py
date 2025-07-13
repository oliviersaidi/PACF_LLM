import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300

# Data
categories = ['Repetitive', 'Code', 'Predictive', 'Random', 'WikiText', 'Natural']
pue_values = [97.7, 97.6, 97.3, 95.9, 88.2, 85.9]
phk_values = [74.5, 86.6, 80.4, 78.0, 21.4, 13.7]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# PUE Bar Chart
bars1 = ax1.bar(categories, pue_values, color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.2)
ax1.set_ylabel('PUE (%)', fontsize=12)
ax1.set_title('Pattern Utilization Efficiency by Category', fontsize=14, pad=20)
ax1.set_ylim(80, 100)
ax1.grid(True, alpha=0.3, linestyle='--')

# Add value labels on bars
for bar, value in zip(bars1, pue_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{value:.1f}', ha='center', va='bottom', fontsize=9)

# Add average line
avg_pue = 93.8
ax1.axhline(y=avg_pue, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Average: {avg_pue}%')
ax1.legend()

# PHK Bar Chart
bars2 = ax2.bar(categories, phk_values, color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.2)
ax2.set_ylabel('PHK (%)', fontsize=12)
ax2.set_title('Pattern Harnessing Coefficient by Category', fontsize=14, pad=20)
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3, linestyle='--')

# Add value labels on bars
for bar, value in zip(bars2, phk_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{value:.1f}', ha='center', va='bottom', fontsize=9)

# Add average line
avg_phk = 59.1
ax2.axhline(y=avg_phk, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Average: {avg_phk}%')
ax2.legend()

# Rotate x-axis labels
ax1.set_xticklabels(categories, rotation=45, ha='right')
ax2.set_xticklabels(categories, rotation=45, ha='right')

plt.tight_layout()
plt.savefig('pacf_pue_phk_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('pacf_pue_phk_comparison.png', dpi=300, bbox_inches='tight')
plt.show()