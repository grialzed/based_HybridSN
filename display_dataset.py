import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import scipy.io as sio

# 读取校正后数据
data_mat = sio.loadmat('data/Indian_pines_corrected.mat')
print(data_mat.keys())  # 先查看变量名

data = data_mat['indian_pines_corrected']   # 形状 (145,145,200)
gt   = sio.loadmat('data/Indian_pines_gt.mat')['indian_pines_gt']  # 形状 (145,145)

data = data.astype(np.float32)
gt   = gt.astype(np.int32)

print('data shape:', data.shape)  # (145, 145, 200)
print('gt shape  :', gt.shape)    # (145, 145)
print('类别标签  :', np.unique(gt))  # 0~16

# 假设 data 形状为 (145,145,200)，gt 形状为 (145,145)
# 选择三个波段做假彩色，例如第30、20、10波段（注意Python索引从0开始）
rgb = data[:, :, [29, 19, 9]].astype(float)
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

# 16类 + 背景，共17种颜色；可按需替换
colors = [
    '#ffffff', '#e6194b', '#3cb44b', '#ffe119', '#4363d8',
    '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c',
    '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',
    '#800000', '#aaffc3'
]
cmap = ListedColormap(colors)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

axes[0].imshow(rgb)
axes[0].set_title('(a) False-color composite')
axes[0].axis('off')

im = axes[1].imshow(gt, cmap=cmap, vmin=0, vmax=16, interpolation='nearest')
axes[1].set_title('(b) Ground truth')
axes[1].axis('off')

class_names = [
    'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
    'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
    'Hay-windrowed', 'Oats', 'Soybean-notill',
    'Soybean-mintill', 'Soybean-clean', 'Wheat',
    'Woods', 'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers'
]

legend_elements = [Patch(facecolor=colors[i+1], label=f'{i+1}. {name}')
                   for i, name in enumerate(class_names)]
axes[1].legend(handles=legend_elements, bbox_to_anchor=(1.05, 0.5),
               loc='center left', fontsize=7, frameon=False)

plt.tight_layout()
plt.savefig('indian_pines_dataset.png', dpi=300, bbox_inches='tight')
plt.show()

import seaborn as sns
from scipy.stats import entropy

# 假设 data 形状为 (145, 145, 200)，已完成读取
H, W, B = data.shape

def band_entropy(band):
    """计算单个波段的香农熵（离散化后）"""
    pixels = band.flatten()
    # 归一化到 [0,1]
    pixels = (pixels - pixels.min()) / (pixels.max() - pixels.min() + 1e-8)
    # 分箱求直方图，避免熵估计受离散化影响过大
    hist, _ = np.histogram(pixels, bins=64, range=(0, 1), density=True)
    hist = hist[hist > 0]
    return entropy(hist)

# ---------- 1. 信息熵：各波段包含多少“不确定度” ----------
entropies = np.array([band_entropy(data[:, :, b]) for b in range(B)])

# ---------- 2. 所选波段与其余波段的相关性 ----------
selected_bands = [9, 19, 29]   # 0-based: 第10、20、30波段
selected_names = ['Band 10', 'Band 20', 'Band 30']

# 将 data 展平为 (像元数, 波段数)
pixels = data.reshape(-1, B)
# 计算所选三个波段与全部波段之间的 Pearson 相关系数
corr_with_all = []
for b in selected_bands:
    corr = np.array([np.corrcoef(pixels[:, b], pixels[:, j])[0, 1] for j in range(B)])
    corr_with_all.append(corr)
corr_with_all = np.array(corr_with_all)  # shape: (3, B)

# 计算所选波段之间的相关矩阵
corr_selected = np.corrcoef(pixels[:, selected_bands].T)

# ---------- 可视化 ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 子图1：全波段信息熵曲线，标出所选三个波段
axes[0].plot(entropies, color='gray', lw=0.8, label='All bands')
for b, name in zip(selected_bands, selected_names):
    axes[0].scatter(b, entropies[b], color='red', zorder=5, s=50)
    axes[0].annotate(name, (b, entropies[b]), textcoords="offset points",
                     xytext=(5, 5), fontsize=8)
axes[0].set_xlabel('Band index (0-based)')
axes[0].set_ylabel('Shannon entropy')
axes[0].set_title('Information content per band')
axes[0].legend()

# 子图2：所选波段与全部波段的相关性曲线
for corr, name, color in zip(corr_with_all, selected_names, ['green', 'blue', 'orange']):
    axes[1].plot(corr, lw=0.8, label=name, color=color, alpha=0.8)
axes[1].axhline(0, color='black', lw=0.5)
axes[1].set_xlabel('Band index (0-based)')
axes[1].set_ylabel('Pearson correlation with selected band')
axes[1].set_title('Correlation profile of selected bands')
axes[1].legend(fontsize=8)

# 子图3：所选三波段之间的相关矩阵
sns.heatmap(corr_selected, annot=True, fmt='.3f', cmap='coolwarm',
            xticklabels=selected_names, yticklabels=selected_names,
            ax=axes[2], vmin=0, vmax=1)
axes[2].set_title('Correlation among the three selected bands')

plt.tight_layout()
plt.savefig('band_selection_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------- 打印关键数值 ----------
print('=== 所选波段的信息熵 ===')
for b, name in zip(selected_bands, selected_names):
    print(f'{name}: entropy = {entropies[b]:.4f}')
print(f'全波段平均熵: {entropies.mean():.4f}')
print(f'\n=== 所选波段之间的相关系数 ===')
print(np.round(corr_selected, 4))
