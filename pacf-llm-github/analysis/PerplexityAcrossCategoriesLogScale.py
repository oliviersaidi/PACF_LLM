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
perplexity_values = [4.0, 3.5, 3.1, 212.3, 9.6, 11.6]

# Figure 2: Perplexity comparison
fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#1f77b4' if p < 50 else '#e74c3c' for p in perplexity_values]

bars = ax.bar(categories, perplexity_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

# Log scale for y-axis due to random outlier
ax.set_yscale('log')
ax.set_ylabel('Perplexity (log scale)', fontsize=12)
ax.set_title('Generation Quality Across Text Categories', fontsize=14, pad=20)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

# Add value labels
for bar, value in zip(bars, perplexity_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
            f'{value:.1f}', ha='center', va='bottom', fontsize=10)

# Add horizontal line for average (excluding random)
avg_perp_no_random = 7.4
ax.axhline(y=avg_perp_no_random, color='green', linestyle='--', linewidth=2, 
           alpha=0.7, label=f'Avg (excl. Random): {avg_perp_no_random:.1f}')

ax.legend()
ax.set_xticklabels(categories, rotation=45, ha='right')

plt.tight_layout()
plt.savefig('pacf_perplexity_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('pacf_perplexity_comparison.png', dpi=300, bbox_inches='tight')
plt.show()