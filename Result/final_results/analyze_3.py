import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# --- 日本語フォント設定 ---
font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc' 
try:
    jp_font_prop = font_manager.FontProperties(fname=font_path)
    print(f"フォント '{font_path}' の読み込みに成功しました。")
except IOError:
    print(f"警告: 日本語フォントパス '{font_path}' が見つかりません。")
    jp_font_prop = font_manager.FontProperties(family='sans-serif')
# -------------------------

try:
    # --- ファイルの読み込み ---
    # ★★★ ご自身の最新のCSVファイル名に書き換えてください ★★★
    file_path = '20250806_044047_scalability_experiment.csv'
    df = pd.read_csv(file_path)
    print(f"'{file_path}' の読み込みに成功しました。")

    # --- データの準備 ---
    df['is_feasible'] = df['PSO_Bottleneck'] >= 0
    df_valid_pso = df[df['is_feasible']].copy()
    df_valid_pso.loc[:, 'Optimality'] = df_valid_pso.apply(
        lambda row: (row['PSO_Bottleneck'] / row['Optimal_Bottleneck']) * 100
        if row['Optimal_Bottleneck'] > 0 else 0,
        axis=1
    )
    
    graph_type_labels = {'random': 'ランダム', 'grid': 'グリッド', 'ba': 'BAモデル'}
    df['graph_type_jp'] = df['graph_type'].map(graph_type_labels)
    df_valid_pso['graph_type_jp'] = df_valid_pso['graph_type'].map(graph_type_labels)

    # --- グラフ描画 ---
    sns.set_theme(style="whitegrid", context="talk")

    # グラフ1: 解の精度 vs ノード数（遅延制約別にプロット）
    g1 = sns.relplot(
        data=df_valid_pso,
        kind='line',
        x='Num_Nodes',
        y='Optimality',
        hue='delay_multiplier',
        col='graph_type_jp',
        col_order=['ランダム', 'グリッド', 'BAモデル'],
        palette='coolwarm',
        marker='o',
        lw=2.5,
        height=6,
        aspect=1.2,
        ci=None
    )
    g1.fig.suptitle('ネットワーク規模とPSOの解の精度（トポロジー・遅延制約別）', y=1.05, fontproperties=jp_font_prop, fontsize=20)
    g1.set_axis_labels("ノード数", "最適度（%）", fontproperties=jp_font_prop, fontsize=14)
    g1.set_titles("{col_name}", fontproperties=jp_font_prop, fontsize=16)
    g1.set(ylim=(0, 105))
    legend1 = g1.legend
    legend1.set_title("遅延倍率", prop=jp_font_prop)
    for text in legend1.get_texts():
        text.set_fontproperties(jp_font_prop)
    plt.savefig('plot_optimality_by_constraint.png')
    print("グラフ1 'plot_optimality_by_constraint.png' を保存しました。")


    # グラフ2: 計算時間 vs ノード数（遅延制約別にプロット）
    df_time = df.rename(columns={'PSO_Time_sec': 'PSO', 'Optimal_Time_sec': '厳密解法'})
    df_time_melted = df_time.melt(
        id_vars=['Num_Nodes', 'delay_multiplier', 'graph_type_jp'], 
        value_vars=['PSO', '厳密解法'],
        var_name='アルゴリズム',
        value_name='計算時間 (秒)'
    )
    
    g2 = sns.relplot(
        data=df_time_melted,
        kind='line',
        x='Num_Nodes',
        y='計算時間 (秒)',
        hue='アルゴリズム',
        style='delay_multiplier',
        col='graph_type_jp',
        col_order=['ランダム', 'グリッド', 'BAモデル'],
        markers=True,
        dashes=True,
        lw=2.5,
        palette={'PSO': 'blue', '厳密解法': 'green'},
        height=6,
        aspect=1.2,
        ci=None
    )
    g2.fig.suptitle('ネットワーク規模と計算時間の関係（トポロジー・遅延制約別）', y=1.05, fontproperties=jp_font_prop, fontsize=20)
    g2.set_axis_labels("ノード数", "平均計算時間（秒）", fontproperties=jp_font_prop, fontsize=14)
    g2.set_titles("{col_name}", fontproperties=jp_font_prop, fontsize=16)
    g2.set(yscale="log")
    legend2 = g2.legend
    legend2.set_title("凡例", prop=jp_font_prop)
    for text in legend2.get_texts():
        text.set_fontproperties(jp_font_prop)
    plt.savefig('plot_time_by_constraint.png')
    print("グラフ2 'plot_time_by_constraint.png' を保存しました。")

except FileNotFoundError:
    print(f"エラー: ファイル '{file_path}' が見つかりません。")
except Exception as e:
    print(f"エラーが発生しました: {e}")