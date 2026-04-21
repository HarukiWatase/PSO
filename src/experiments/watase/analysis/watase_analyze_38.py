import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import japanize_matplotlib
import os
import glob
from matplotlib.patches import Patch

# ==========================================
# 設定
# ==========================================
TARGET_DIR = "Result/parallel_gen_logs"
SAVE_PREFIX = "parallel_gen_graph_range_fixed" # ファイル名変更

# ★★★ 変更点: 軸の表示範囲を指定 ★★★
X_LIMIT = 100  # 横軸（世代）の最大値
Y_LIMIT = 50   # 縦軸（ボトルネック帯域）の最大値
# ==========================================

# ==========================================
# データ読み込み
# ==========================================
list_of_sum = glob.glob(f'{TARGET_DIR}/*_summary.csv')
list_of_log = glob.glob(f'{TARGET_DIR}/*_generation_log.csv')

if not list_of_sum or not list_of_log:
    print("エラー: ログファイルが見つかりません。")
    exit()

summary_csv = max(list_of_sum, key=os.path.getctime)
gen_log_csv = max(list_of_log, key=os.path.getctime)

print(f"Summary: {summary_csv}")
print(f"Gen Log: {gen_log_csv}")

df_summary = pd.read_csv(summary_csv)
df_gen_log = pd.read_csv(gen_log_csv)

# 厳密解法の平均計算時間とBN
exact_avg_time = df_summary['Exact_Time'].mean()
exact_avg_bn = df_summary['Exact_BN'].mean()

# PSOデータの整形
df_gen_log['Bottleneck'] = df_gen_log['Bottleneck'].replace(-1, 0)
avg_bn_per_gen = df_gen_log.groupby('Generation')['Bottleneck'].mean()
avg_time_per_gen = df_gen_log.groupby('Generation')['Time'].mean()

# 推測世代の計算
if exact_avg_time <= 0:
    inferred_gen = 0
    display_label = "厳密解法: 計測不能"
    line_color = "gray"
elif exact_avg_time > avg_time_per_gen.max():
    inferred_gen = avg_time_per_gen.index.max()
    display_label = f"厳密解法 (範囲外: {exact_avg_time:.1f}s)"
    line_color = "purple"
else:
    inferred_gen = avg_time_per_gen[avg_time_per_gen >= exact_avg_time].index.min()
    display_label = f"厳密解法の平均時間 ({exact_avg_time:.1f}s)"
    line_color = "#d62728"

# --- グラフ描画 ---
plt.figure(figsize=(10, 6))
plt.grid(False) # グリッドなし

# PSOプロット
plt.plot(
    avg_bn_per_gen.index, 
    avg_bn_per_gen.values, 
    label='提案手法 (並列PSO)', 
    color='#1f77b4', 
    linewidth=1.5,
    marker='o',
    markersize=3
)

# 厳密解法の垂直線
plt.axvline(
    x=inferred_gen, 
    color=line_color, 
    linestyle='--', 
    linewidth=1,
    label=display_label
)

# 厳密解法の×印
plot_bn = exact_avg_bn if exact_avg_bn > 0 else 0
plt.scatter(
    [inferred_gen], 
    [plot_bn], 
    color=line_color, 
    marker='X', 
    s=50, 
    zorder=10,
    label=f'厳密解法 (BN: {exact_avg_bn:.1f})'
)

# ★★★ 変更点: 軸の範囲指定 ★★★
plt.xlim(left=0, right=X_LIMIT) 
plt.ylim(bottom=0, top=Y_LIMIT)

plt.xlabel('世代 (Generation)', fontsize=16)
plt.ylabel('ボトルネック帯域 (Mbps)', fontsize=16)
plt.legend(fontsize=12, loc='lower right')

# 枠線を整える
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)

plt.tight_layout()
plt.savefig(f'{SAVE_PREFIX}.png', dpi=300)
print(f"グラフ保存完了: {SAVE_PREFIX}.png")
plt.show()