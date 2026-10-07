import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置学术绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# ================= 1. 绘制四象限分布图 =================
labels = ['Quad 0\n(Strongly Rec. KAN)', 'Quad 1\n(Rec. MLP - Latency)',
          'Quad 2\n(Strongly Rec. MLP)', 'Quad 3\n(Rec. KAN - Anomaly)']
counts = [2555, 12, 4929, 4]

plt.figure(figsize=(8, 5))
ax = sns.barplot(x=labels, y=counts, palette="viridis")
plt.yscale('log') # 使用对数坐标，因为 12 和 4 太小了
plt.title('Distribution of 7,500 Datasets Across Diagnostic Quadrants', fontsize=14, pad=15)
plt.ylabel('Number of Datasets (Log Scale)', fontsize=12)

# 在柱子上添加具体数字
for i, p in enumerate(ax.patches):
    ax.annotate(f'{counts[i]}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=12, xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig('quadrant_dist.pdf', dpi=300, bbox_inches='tight')
plt.show()

# ================= 2. 绘制特征重要性条形图 =================
features = [
    'task_type_code', 'n_samples', 'n_features', 'intrinsic_dim',
    'sparsity', 'samples_per_feature', 'smoothness_proxy', 'kurtosis', 'skewness'
]
importances = [20.62, 15.60, 12.04, 10.92, 9.77, 8.37, 8.30, 7.73, 6.66]

plt.figure(figsize=(10, 6))
# 颜色渐变
ax = sns.barplot(x=importances, y=features, palette="rocket")
plt.title('XGBoost Diagnoser: Feature Importance', fontsize=14, pad=15)
plt.xlabel('Importance (%)', fontsize=12)
plt.ylabel('Meta-Features', fontsize=12)

# 在条形图末尾添加百分比文本
for i, p in enumerate(ax.patches):
    width = p.get_width()
    plt.text(width + 0.3, p.get_y() + p.get_height()/2. + 0.1, f'{importances[i]:.2f}%', ha='left', fontsize=11)

plt.xlim(0, 25)
plt.tight_layout()
plt.savefig('feature_importance.pdf', dpi=300, bbox_inches='tight')
plt.show()