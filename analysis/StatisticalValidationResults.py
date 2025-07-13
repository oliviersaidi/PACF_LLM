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

# Figure 5: Statistical validation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Validation metrics
validation_metrics = ['PUE', 'PHK', 'Perplexity', 'Speed']
validation_means = [87.2, 10.4, 6.93, 10.7]
validation_stds = [0.19, 2.46, 0.02, 1.6]

# Error bar plot
x = np.arange(len(validation_metrics))
bars = ax1.bar(x, validation_means, yerr=validation_stds, capsize=10, 
                color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'], 
                alpha=0.8, edgecolor='black', linewidth=1.2)

ax1.set_ylabel('Value', fontsize=12)
ax1.set_title('Statistical Validation Results (30 runs)', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(validation_metrics)
ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

# Add value labels
for bar, mean, std in zip(bars, validation_means, validation_stds):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + std + 1,
            f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=9)

# P-values visualization
p_values = [1.03e-6, 4.02e-9]
p_labels = ['Speed', 'Perplexity']
bars2 = ax2.bar(p_labels, [-np.log10(p) for p in p_values], 
                 color=['#e74c3c', '#27ae60'], alpha=0.8, edgecolor='black', linewidth=1.2)

ax2.set_ylabel('-log10(p-value)', fontsize=12)
ax2.set_title('Statistical Significance of Results', fontsize=14)
ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
ax2.axhline(y=-np.log10(0.001), color='orange', linestyle='--', label='p=0.001')
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
ax2.legend()

# Add actual p-values as labels
for bar, p_val in zip(bars2, p_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
            f'p={p_val:.2e}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('pacf_statistical_validation.pdf', dpi=300, bbox_inches='tight')
plt.savefig('pacf_statistical_validation.png', dpi=300, bbox_inches='tight')
plt.show()