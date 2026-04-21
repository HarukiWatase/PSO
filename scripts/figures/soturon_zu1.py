import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import matplotlib.patches as patches
import japanize_matplotlib

# ==========================================
# ★設定: 論文用 (IEICE等) 最終完成版 v11 (キャプション下) ★
# ==========================================
plt.rcParams['svg.fonttype'] = 'none'
# plt.rcParams['font.family'] = 'sans-serif' 
plt.rcParams['font.size'] = 23          # フォントサイズ維持
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.linewidth'] = 2.5

# 図のサイズ設定
fig, axes = plt.subplots(1, 2, figsize=(18, 8), # ★高さを少し増やしました(7->8) 下の文字スペース確保のため
                         gridspec_kw={'width_ratios': [1, 1.8], 'wspace': 0.1}) 
ax1, ax2 = axes

# ==========================================
# 関数: 等高線描画
# ==========================================
def plot_contours(ax, center_x, center_y, mode='simple'):
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()
    X, Y = np.meshgrid(np.linspace(x_range[0], x_range[1], 100),
                       np.linspace(y_range[0], y_range[1], 100))
    
    if mode == 'simple':
        Z = -np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        levels = 8
        alpha_val = 0.5
    elif mode == 'double_well':
        Z1 = 1.2 * np.exp(-((X - 4.0)**2 + (Y - 5.0)**2) / 5.0) 
        Z2 = 3.5 * np.exp(-((X - 12.0)**2 + (Y - 5.0)**2) / 6.0) 
        Z = Z1 + Z2
        levels = np.linspace(0.1, 3.0, 9) 
        alpha_val = 0.45

    ax.contour(X, Y, Z, levels=levels, colors='silver', linewidths=2.0, alpha=alpha_val, linestyles='-', zorder=0)

# ==========================================
# 左側: (a) 空間的 lBest
# ==========================================
center_x, center_y = 5.0, 5.0
ax1.set_xlim(3.2, 6.8); ax1.set_ylim(3.2, 6.8); ax1.set_aspect('equal')

# ★変更点: set_title を set_xlabel に変更し、labelpadで余白調整
# ax1.set_title(...) は削除
ax1.set_xlabel("(a) 空間的 lBest トポロジー", fontsize=25, labelpad=20, fontweight='bold')

ax1.set_xticks([]); ax1.set_yticks([])
for spine in ax1.spines.values(): spine.set_color('gray')

plot_contours(ax1, center_x, center_y, mode='simple')

# 緑粒子の配置
r_neighbor = 1.1 
angles = [np.pi, 3*np.pi/4, np.pi/4, np.pi/30] # 左, 左上, 右上, 右
gx = [center_x + r_neighbor * np.cos(a) for a in angles]
gy = [center_y + r_neighbor * np.sin(a) for a in angles]
radius = r_neighbor + 0.25

# 白粒子の生成 (緑範囲外)
np.random.seed(42)
wx, wy = [], []
while len(wx) < 10:
    tmp_x = np.random.uniform(2.5, 7.5)
    tmp_y = np.random.uniform(2.5, 7.5)
    dist = np.sqrt((tmp_x - center_x)**2 + (tmp_y - center_y)**2)
    if dist > radius + 0.1:
        wx.append(tmp_x); wy.append(tmp_y)

# 描画
ax1.add_patch(patches.Circle((center_x, center_y), radius, fill=True, color='limegreen', alpha=0.1, zorder=1))
for i in range(len(gx)):
    ax1.add_patch(patches.FancyArrowPatch((gx[i], gy[i]), (center_x, center_y),
        arrowstyle='->', mutation_scale=35, color='limegreen', alpha=1.0, linewidth=4.0, zorder=10, shrinkB=20))
ax1.scatter(wx, wy, c='white', s=500, edgecolors='gray', linewidth=2.5, alpha=0.9, zorder=3)
ax1.scatter(gx, gy, c='limegreen', s=700, edgecolors='darkgreen', linewidth=3, zorder=4)
ax1.scatter(center_x, center_y, c='blue', s=900, edgecolors='black', linewidth=3, zorder=5)

ax1.text(center_x, center_y - 0.55, "lBest粒子", ha='center', va='top', fontsize=23, fontweight='bold', color='navy', zorder=20, linespacing=1.1)
ax1.text(center_x, center_y + radius + 0.05, "近傍粒子", ha='center', fontsize=23, fontweight='bold', color='darkgreen', zorder=20)


# ==========================================
# 右側: (b) Restart 戦略
# ==========================================
# ★変更点: set_title を set_xlabel に変更
# ax2.set_title(...) は削除
ax2.set_xlabel("(b) リスタート戦略", fontsize=25, labelpad=20, fontweight='bold')

ax2.set_xlim(0, 16); ax2.set_ylim(0, 10); ax2.set_aspect('equal')
ax2.set_xticks([]); ax2.set_yticks([])
for spine in ax2.spines.values(): 
    spine.set_linestyle('-')  # 実線に戻す
    spine.set_color('gray')   # 色を(a)に合わせる
plot_contours(ax2, 0, 0, mode='double_well')

# Left: Before
cx1, cy1 = 4.0, 5.0
sx = np.random.normal(cx1, 0.3, 10); sy = np.random.normal(cy1, 0.3, 10)
ax2.scatter(sx, sy, c='salmon', s=500, marker='o', alpha=1.0, edgecolors='darkred', linewidth=2.5, zorder=3)

# テキスト位置維持 (3.2)
ax2.text(cx1, 3.5, "① 停滞検知", ha='center', va='top', fontsize=23, color='darkred', fontweight='bold')
ax2.add_patch(patches.FancyArrowPatch((6.0, 5), (9.0, 5), mutation_scale=40, color='gray', arrowstyle='->', lw=5, zorder=1))

# Right: After
cx2, cy2 = 12.0, 5.0

# エリート配置
ax2.scatter(cx2, cy2, c='gold', s=1200, marker='*', edgecolors='black', linewidth=2.5, zorder=10)
ax2.text(cx2, 4.1, "エリート\n粒子保存", ha='center', va='top', fontsize=21, color='black', fontweight='bold', linespacing=1.1, zorder=20)

# 青粒子の生成（テキスト領域回避）
rx, ry = [], []
np.random.seed(42)
text_avoid_x_min, text_avoid_x_max = 10.5, 13.5
text_avoid_y_min, text_avoid_y_max = 2.5, 4.3

while len(rx) < 10:
    angle = np.random.uniform(0, 2 * np.pi)
    r = np.random.uniform(2.0, 3.5)
    cand_x = cx2 + r * np.cos(angle)
    cand_y = cy2 + r * np.sin(angle)
    if (text_avoid_x_min < cand_x < text_avoid_x_max) and (text_avoid_y_min < cand_y < text_avoid_y_max):
        continue
    rx.append(cand_x); ry.append(cand_y)

# 描画
ax2.scatter(rx, ry, c='skyblue', s=500, marker='o', alpha=0.9, edgecolors='blue', linewidth=2.5, zorder=3)

# テキスト位置維持 (11.5, 8.7)
ax2.text(11.8, 8.5, "② 再初期化", ha='center', va='bottom', fontsize=23, color='navy', fontweight='bold')

plt.tight_layout()
output_dir = Path("assets/figures/root")
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "svg/fig1_final_v11_bottom.svg", format='svg', bbox_inches='tight', pad_inches=0.1)
plt.savefig(output_dir / "png/fig1_final_v11_bottom.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
print("保存完了: assets/figures/root/png/fig1_final_v11_bottom.png")
plt.show()