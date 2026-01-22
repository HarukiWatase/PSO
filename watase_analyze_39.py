import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np
import japanize_matplotlib

# ==========================================
# 設定
# ==========================================
TARGET_DIR = "Result/feasibility_benchmark_pareto"
SAVE_PREFIX_RATE = "feasibility_rate_result_fix"
SAVE_PREFIX_PARETO = "pareto_count_analysis_fix"

# ★ここが重要: グラフに表示する並び順を固定します
# 以前: アルファベット順 (Global -> Restart -> Spatial) になっていた可能性大
# 修正: Global -> Spatial -> Restart の順にします
METHOD_ORDER = ['Global', 'Spatial', 'Restart']

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

# 成功率計算用にTrue/Falseを1/0に変換
df['Is_Success_Num'] = df['Is_Success'].astype(int)

# ==========================================
# グラフ1: 遅延倍率 vs 成功率 (Bar Plot)
# ==========================================
plt.figure(figsize=(10, 6))
plt.grid(False)

# 集計: 倍率と手法ごとの成功率平均
success_rates = df.groupby(['Multiplier', 'Method'])['Is_Success_Num'].mean().reset_index()
success_rates['Success_Rate_Percent'] = success_rates['Is_Success_Num'] * 100

ax = sns.barplot(
    data=success_rates,
    x='Multiplier',
    y='Success_Rate_Percent',
    hue='Method',
    hue_order=METHOD_ORDER, # ★ここで順序を指定
    palette='viridis',
    edgecolor='black',
    linewidth=1
)

plt.xlabel('遅延制約倍率 (小さいほど厳しい)', fontsize=16)
plt.ylabel('成功率 (%)', fontsize=16)
plt.ylim(0, 105)

plt.tick_params(axis='both', labelsize=12)

# 数値を棒グラフの上に表示
#for container in ax.containers:
#    ax.bar_label(container, fmt='%.0f%%', fontsize=11, padding=3)

# ▼▼▼▼▼▼▼ 修正箇所 ▼▼▼▼▼▼▼

# 1. 現在の凡例の情報（色や形）を取得
handles, _ = ax.get_legend_handles_labels()

# 2. 変更したい名前をリストで定義
# ★重要: METHOD_ORDER = ['Global', 'Spatial', 'Restart'] の順序に合わせて書いてください
new_labels = ['Global PSO', 'Local PSO', 'Restart PSO'] 

# 3. handlesと一緒に新しいラベルを渡して凡例を再描画
plt.legend(handles, new_labels, title='手法', fontsize=16, title_fontsize=16, loc='lower right')

# ▲▲▲▲▲▲▲ 修正箇所終わり ▲▲▲▲▲▲▲
plt.tight_layout()
plt.savefig(f'{SAVE_PREFIX_RATE}.png', dpi=300)
print(f"グラフ保存: {SAVE_PREFIX_RATE}.png")
plt.show()

# ==========================================
# グラフ2: 遅延倍率 vs パレート最適解の数 (Line Plot)
# ==========================================
df_pareto = df[['Multiplier', 'Trial', 'Pareto_Count']].drop_duplicates()
pareto_means = df_pareto.groupby('Multiplier')['Pareto_Count'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.grid(False)

# パレート数の推移（左軸）
ax1 = plt.gca()
line1 = ax1.plot(
    pareto_means['Multiplier'].astype(str),
    pareto_means['Pareto_Count'],
    marker='o',
    color='tab:red',
    linewidth=2,
    markersize=8,
    label='平均パレート解数'
)
ax1.set_xlabel('遅延制約倍率', fontsize=16)
ax1.set_ylabel('パレート最適経路の数 (個)', fontsize=16, color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')

# 参考としてRestart手法の成功率を右軸に重ねる
restart_success = success_rates[success_rates['Method'] == 'Restart']
ax2 = ax1.twinx()
line2 = ax2.plot(
    restart_success['Multiplier'].astype(str),
    restart_success['Success_Rate_Percent'],
    marker='s',
    color='tab:green',
    linewidth=2,
    linestyle='--',
    markersize=8,
    label='Restart手法の成功率'
)
ax2.set_ylabel('成功率 (%)', fontsize=16, color='tab:green')
ax2.tick_params(axis='y', labelcolor='tab:green')
ax2.set_ylim(0, 110)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', fontsize=12)

plt.tight_layout()
plt.savefig(f'{SAVE_PREFIX_PARETO}.png', dpi=300)
print(f"グラフ保存: {SAVE_PREFIX_PARETO}.png")
plt.show()