import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# --- 日本語フォント設定 ---
# 以前のやり取りで確認した、ご自身のPC上の正しいフォントファイルのパス
font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W0.ttc' 
try:
    jp_font_prop = font_manager.FontProperties(fname=font_path)
    print(f"フォント '{font_path}' の読み込みに成功しました。")
except IOError:
    print(f"警告: 日本語フォントパスが見つかりません。")
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

    # グラフ1: スケーラビリティ評価（計算時間）
    # データをグラフ化しやすいように整形（縦持ちデータに変換）
    df_time = df.rename(columns={'PSO_Time_sec': 'PSO', 'Optimal_Time_sec': '厳密解法'})
    df_time_melted = df_time.melt(
        id_vars=['Num_Nodes', 'graph_type_jp'], 
        value_vars=['PSO', '厳密解法'],
        var_name='アルゴリズム',
        value_name='計算時間 (秒)'
    )

    plt.figure(figsize=(12, 8))
    ax1 = sns.lineplot(
        data=df_time_melted, 
        x='Num_Nodes', 
        y='計算時間 (秒)', 
        hue='graph_type_jp',
        style='アルゴリズム', # アルゴリズムの種類で線のスタイル（実線/破線）を変える
        markers=True, 
        dashes=True, 
        lw=2.5
    )
    ax1.set_title('ネットワーク規模と計算時間の関係', fontproperties=jp_font_prop, fontsize=20, pad=20)
    ax1.set_xlabel('ノード数', fontproperties=jp_font_prop, fontsize=14)
    ax1.set_ylabel('平均計算時間（秒）※対数スケール', fontproperties=jp_font_prop, fontsize=14)
    ax1.set_yscale('log')
    handles, labels = ax1.get_legend_handles_labels()
    legend1 = ax1.legend(handles=handles, labels=labels, prop=jp_font_prop)
    legend1.get_title().set_fontproperties(jp_font_prop)
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('plot_scalability_time.png')
    print("グラフ1 'plot_scalability_time.png' を保存しました。")


    # グラフ2: スケーラビリティ評価（解の精度）
    plt.figure(figsize=(12, 8))
    ax2 = sns.lineplot(data=df_valid_pso, x='Num_Nodes', y='Optimality', hue='graph_type_jp', 
                       style='graph_type_jp', markers=True, dashes=False, lw=2.5)
    ax2.set_title('ネットワーク規模とPSOの解の精度', fontproperties=jp_font_prop, fontsize=20, pad=20)
    ax2.set_xlabel('ノード数', fontproperties=jp_font_prop, fontsize=14)
    ax2.set_ylabel('最適度（%）', fontproperties=jp_font_prop, fontsize=14)
    ax2.set_ylim(0, 101) # 精度が高い領域にズーム
    handles, labels = ax2.get_legend_handles_labels()
    legend2 = ax2.legend(handles=handles, labels=labels, title='グラフ種類', prop=jp_font_prop)
    legend2.get_title().set_fontproperties(jp_font_prop)
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('plot_scalability_optimality.png')
    print("グラフ2 'plot_scalability_optimality.png' を保存しました。")

    # グラフ3: 制約の厳しさの影響評価（実行可能解 発見率）
    feasibility_rate = df.groupby(['graph_type_jp', 'delay_multiplier'])['is_feasible'].mean().reset_index()
    feasibility_rate['is_feasible'] *= 100

    plt.figure(figsize=(12, 8))
    ax3 = sns.pointplot(data=feasibility_rate, x='delay_multiplier', y='is_feasible', hue='graph_type_jp',
                        hue_order=['ランダム', 'グリッド', 'BAモデル'], markers='o', dodge=True)
    ax3.set_title('遅延制約の厳しさと実行可能解の発見率', fontproperties=jp_font_prop, fontsize=20, pad=20)
    ax3.set_xlabel('遅延倍率（値が小さいほど制約が厳しい）', fontproperties=jp_font_prop, fontsize=14)
    ax3.set_ylabel('実行可能解 発見率（%）', fontproperties=jp_font_prop, fontsize=14)
    ax3.set_ylim(0, 110)
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles=handles, labels=labels, title='グラフ種類', prop=jp_font_prop, title_fontproperties=jp_font_prop)
    ax3.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('plot_constraint_feasibility.png')
    print("グラフ3 'plot_constraint_feasibility.png' を保存しました。")

except FileNotFoundError:
    print(f"エラー: ファイル '{file_path}' が見つかりません。")
except Exception as e:
    print(f"エラーが発生しました: {e}")