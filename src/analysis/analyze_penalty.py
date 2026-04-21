import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# --- 日本語フォント設定 ---
# ご自身の環境で確認済みの、正しいフォントファイルのパスを指定してください。
font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc' 
try:
    jp_font_prop = font_manager.FontProperties(fname=font_path)
    print(f"フォント '{font_path}' の読み込みに成功しました。")
except IOError:
    print(f"警告: 指定された日本語フォントパス '{font_path}' が見つかりません。")
    jp_font_prop = font_manager.FontProperties(family='sans-serif')
# -------------------------

try:
    # --- ファイルの読み込み ---
    # ★★★ ご自身の最新のCSVファイル名に書き換えてください ★★★
    file_path = '20250723_175402_pso_comparison.csv' 
    df = pd.read_csv(file_path)
    print(f"'{file_path}' の読み込みに成功しました。")

    # --- データの準備 ---
    # is_feasible: PSOが有効な解（gBest_feasible_bn が初期値-1.0から更新されたか）を見つけたかどうかのフラグ
    df['is_feasible'] = df['PSO_Bottleneck'] >= 0
    
    # 最適度（最適解に対する達成率 %）を計算
    df_valid_pso = df[df['is_feasible']].copy()
    df_valid_pso.loc[:, 'Optimality'] = df_valid_pso.apply(
        lambda row: (row['PSO_Bottleneck'] / row['Optimal_Bottleneck']) * 100
        if row['Optimal_Bottleneck'] > 0 else 0,
        axis=1
    )

    # --- グラフ描画 ---
    sns.set_theme(style="whitegrid")
    
    # グラフの凡例と軸ラベルの日本語設定
    pso_version_labels = {'death_penalty': '死のペナルティ', 'penalty_method': 'ペナルティ法'}
    graph_type_labels = {'random': 'ランダム', 'grid': 'グリッド', 'ba': 'BAモデル'}
    df['pso_version_jp'] = df['pso_version'].map(pso_version_labels)
    df['graph_type_jp'] = df['graph_type'].map(graph_type_labels)
    df_valid_pso['pso_version_jp'] = df_valid_pso['pso_version'].map(pso_version_labels)
    df_valid_pso['graph_type_jp'] = df_valid_pso['graph_type'].map(graph_type_labels)

    # グラフ1: 実行可能解 発見率の比較
    feasibility_rate = df.groupby(['graph_type_jp', 'delay_multiplier', 'pso_version_jp'])['is_feasible'].mean().reset_index()
    feasibility_rate['is_feasible'] *= 100

    g1 = sns.catplot(
        data=feasibility_rate, kind="bar",
        x="graph_type_jp", y="is_feasible", hue="pso_version_jp",
        col="delay_multiplier",
        height=6, aspect=1.1,
        order=['ランダム', 'グリッド', 'BAモデル'],
        hue_order=['死のペナルティ', 'ペナルティ法']
    )
    g1.fig.suptitle('PSOバージョン別 実行可能解の発見率', y=1.03, fontproperties=jp_font_prop, fontsize=18)
    g1.set_axis_labels("グラフのトポロジー", "実行可能解 発見率 (%)", fontproperties=jp_font_prop, fontsize=12)
    g1.set_titles("遅延倍率 = {col_name}", fontproperties=jp_font_prop, fontsize=14)
    for ax in g1.axes.flat:
        for label in ax.get_xticklabels():
            label.set_fontproperties(jp_font_prop)
    legend1 = g1.legend
    legend1.set_title("PSOバージョン", prop=jp_font_prop)
    for text in legend1.get_texts():
        text.set_fontproperties(jp_font_prop)
    
    plt.savefig('plot_feasibility_rate.png')
    print("グラフ1 'plot_feasibility_rate.png' を保存しました。")

    # グラフ2: 解の精度（最適度）の比較
    g2 = sns.catplot(
        data=df_valid_pso, kind="box",
        x="graph_type_jp", y="Optimality", hue="pso_version_jp",
        col="delay_multiplier",
        height=6, aspect=1.1,
        order=['ランダム', 'グリッド', 'BAモデル'],
        hue_order=['死のペナルティ', 'ペナルティ法']
    )
    g2.fig.suptitle('PSOバージョン別 解の精度（実行可能解のみ）', y=1.03, fontproperties=jp_font_prop, fontsize=18)
    g2.set_axis_labels("グラフのトポロジー", "最適度 (%)", fontproperties=jp_font_prop, fontsize=12)
    g2.set_titles("遅延倍率 = {col_name}", fontproperties=jp_font_prop, fontsize=14)
    g2.set(ylim=(0, 110))
    for ax in g2.axes.flat:
        for label in ax.get_xticklabels():
            label.set_fontproperties(jp_font_prop)
    legend2 = g2.legend
    legend2.set_title("PSOバージョン", prop=jp_font_prop)
    for text in legend2.get_texts():
        text.set_fontproperties(jp_font_prop)

    plt.savefig('plot_optimality_comparison.png')
    print("グラフ2 'plot_optimality_comparison.png' を保存しました。")

except FileNotFoundError:
    print(f"エラー: ファイル '{file_path}' が見つかりません。")
except Exception as e:
    print(f"エラーが発生しました: {e}")