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

# Data from your evaluation
categories = ['Repetitive', 'Code', 'Predictive', 'Random', 'WikiText', 'Natural']
pue_values = [97.7, 97.6, 97.3, 95.9, 88.2, 85.9]
phk_values = [74.5, 86.6, 80.4, 78.0, 21.4, 13.7]

# Figure 3: PUE vs PHK Relationship
fig, ax = plt.subplots(figsize=(10, 8))

# Create scatter plot
scatter = ax.scatter(phk_values, pue_values, s=200, alpha=0.6, edgecolors='black', linewidth=2)

# Color by category type
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#34495e']
for i, (cat, phk, pue, color) in enumerate(zip(categories, phk_values, pue_values, colors)):
    ax.scatter(phk, pue, s=300, color=color, edgecolors='black', linewidth=2, label=cat)
    # Add category labels
    ax.annotate(cat, (phk, pue), xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('PHK (%)', fontsize=12)
ax.set_ylabel('PUE (%)', fontsize=12)
ax.set_title('Pattern Utilization vs Pattern Harnessing Across Categories', fontsize=14, pad=20)
ax.grid(True, alpha=0.3, linestyle='--')

# Add quadrant lines
ax.axvline(x=59.1, color='red', linestyle='--', alpha=0.5, label='Avg PHK')
ax.axhline(y=93.8, color='red', linestyle='--', alpha=0.5, label='Avg PUE')

# Set axis limits
ax.set_xlim(0, 100)
ax.set_ylim(80, 100)

ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('pacf_pue_phk_scatter.pdf', dpi=300, bbox_inches='tight')
plt.savefig('pacf_pue_phk_scatter.png', dpi=300, bbox_inches='tight')
plt.show()