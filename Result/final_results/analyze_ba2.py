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
    file_path = '20250811_220429_scalability_experiment.csv'
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

    # --- グラフ描画 ---
    sns.set_theme(style="whitegrid", context="talk")

    # グラフ1: 解の精度 vs ノード数
    plt.figure(figsize=(12, 8))
    ax1 = sns.lineplot(
        data=df_valid_pso, 
        x='Num_Nodes', 
        y='Optimality',
        hue='delay_multiplier',
        palette='viridis',
        marker='o', 
        lw=2.5,
        ci=None  # ★★★ 変更点：信頼区間を非表示にする ★★★
    )
    ax1.set_title('ネットワーク規模とPSOの解の精度（BAモデル・遅延制約別）', fontproperties=jp_font_prop, fontsize=20, pad=20)
    ax1.set_xlabel('ノード数', fontproperties=jp_font_prop, fontsize=14)
    ax1.set_ylabel('最適度（%）', fontproperties=jp_font_prop, fontsize=14)
    ax1.set_ylim(0, 105)
    handles, labels = ax1.get_legend_handles_labels()
    legend1 = ax1.legend(handles=handles, labels=labels, title='遅延倍率', prop=jp_font_prop, title_fontproperties=jp_font_prop)
    plt.savefig('ba_newpenalty.png')
    print("グラフ1 'plot_ba_optimality_vs_nodes.png' を保存しました。")


    # グラフ2: 計算時間 vs ノード数
    df_time = df.rename(columns={'PSO_Time_sec': 'PSO', 'Optimal_Time_sec': '厳密解法'})
    df_time_melted = df_time.melt(
        id_vars=['Num_Nodes', 'delay_multiplier'], 
        value_vars=['PSO', '厳密解法'],
        var_name='アルゴリズム',
        value_name='計算時間 (秒)'
    )
    
    plt.figure(figsize=(12, 8))
    ax2 = sns.lineplot(
        data=df_time_melted,
        x='Num_Nodes',
        y='計算時間 (秒)',
        hue='アルゴリズム',
        style='delay_multiplier',
        markers=True,
        dashes=True,
        lw=2.5,
        palette={'PSO': 'blue', '厳密解法': 'green'},
        ci=None  # ★★★ 変更点：信頼区間を非表示にする ★★★
    )
    ax2.set_title('ネットワーク規模と計算時間の関係（BAモデル・遅延制約別）', fontproperties=jp_font_prop, fontsize=20, pad=20)
    ax2.set_xlabel('ノード数', fontproperties=jp_font_prop, fontsize=14)
    ax2.set_ylabel('平均計算時間（秒）※対数スケール', fontproperties=jp_font_prop, fontsize=14)
    ax2.set_yscale('log')
    handles, labels = ax2.get_legend_handles_labels()
    legend2 = ax2.legend(handles=handles, labels=labels, prop=jp_font_prop)
    legend2.get_title().set_fontproperties(jp_font_prop)
    plt.savefig('ba_newpena_time.png')
    print("グラフ2 'plot_ba_time_vs_nodes.png' を保存しました。")

except FileNotFoundError:
    print(f"エラー: ファイル '{file_path}' が見つかりません。")
except Exception as e:
    print(f"エラーが発生しました: {e}")