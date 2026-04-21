import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import japanize_matplotlib

plt.rcParams['axes.unicode_minus'] = False 

# ==========================================
# 設定
# ==========================================
TARGET_DIR = "Result/scalability_benchmark_3methods"
SAVE_PREFIX = "scalability_result_jp_light"  # ファイル名変更
BAND_ALPHA = 0.1  # ★ここで帯の薄さを調整します（0.0〜1.0）。0.1はかなり薄め。

# ==========================================
# データ読み込み
# ==========================================
list_of_files = glob.glob(f'{TARGET_DIR}/*.csv')
if not list_of_files:
    print(f"エラー: ディレクトリ '{TARGET_DIR}' が見つかりません。")
    exit()

latest_file = max(list_of_files, key=os.path.getctime)
print(f"読み込み中: {latest_file}")
df = pd.read_csv(latest_file)

# ==========================================
# スタイル設定
# ==========================================
plt.rcParams['font.size'] = 14

# ==========================================
# グラフ1: 計算時間 vs コア数
# ==========================================
plt.figure(figsize=(10, 6))
plt.grid(False)

ax = sns.lineplot(
    data=df, 
    x='CoreCount', 
    y='Time', 
    hue='Method', 
    style='Method', 
    markers=True, 
    dashes=False,
    linewidth=2.5,
    markersize=10,
    err_style='band',
    ci=95,
    # ★ ここで帯の透明度を指定
    err_kws={'alpha': BAND_ALPHA} 
)

plt.xlabel('コア数 (並列プロセス数)', fontsize=16)
plt.ylabel('平均計算時間 (秒)', fontsize=16)
plt.xticks(sorted(df['CoreCount'].unique()))

# 凡例
handles, labels = ax.get_legend_handles_labels()
# ★ 凡例の四角形もグラフに合わせて薄くする
ci_patch = Patch(facecolor='gray', alpha=BAND_ALPHA, label='95%信頼区間')
handles.append(ci_patch)
plt.legend(handles=handles, title='手法 / 情報', fontsize=12, title_fontsize=12)



plt.tight_layout()
plt.savefig(f'{SAVE_PREFIX}_time.png', dpi=300)
print(f"保存しました: {SAVE_PREFIX}_time.png")
plt.show()

# ==========================================
# グラフ2: 速度向上率 (Speedup)
# ==========================================
baseline_times = df[df['CoreCount'] == 1].groupby('Method')['Time'].mean()
def calculate_speedup(row):
    base = baseline_times.get(row['Method'])
    if base:
        return base / row['Time']
    return np.nan
df['Speedup'] = df.apply(calculate_speedup, axis=1)

plt.figure(figsize=(10, 6))
plt.grid(False)

ax2 = sns.lineplot(
    data=df, 
    x='CoreCount', 
    y='Speedup', 
    hue='Method', 
    style='Method', 
    markers=True, 
    dashes=False,
    linewidth=2.5,
    markersize=10,
    ci=95,
    # ★ ここで帯の透明度を指定
    err_kws={'alpha': BAND_ALPHA}
)

max_core = df['CoreCount'].max()
ideal_line_handle = plt.plot([1, max_core], [1, max_core], 'k--', label='理想的な線形向上', alpha=0.6)[0]

plt.title('並列化による速度向上率', fontsize=18)
plt.xlabel('コア数 (並列プロセス数)', fontsize=16)
plt.ylabel('速度向上率 (1コア比)', fontsize=16)
plt.xticks(sorted(df['CoreCount'].unique()))

# 凡例
handles2, labels2 = ax2.get_legend_handles_labels()
final_handles = handles2 + [ideal_line_handle, ci_patch]
plt.legend(handles=final_handles, title='凡例', fontsize=12)

plt.tight_layout()
plt.savefig(f'{SAVE_PREFIX}_speedup.png', dpi=300)
print(f"保存しました: {SAVE_PREFIX}_speedup.png")
plt.show()