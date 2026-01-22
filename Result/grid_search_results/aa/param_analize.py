import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import glob

# --- 日本語フォント設定 ---
font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'
try:
    font_prop_title = font_manager.FontProperties(fname=font_path, size=22)
    font_prop_subtitle = font_manager.FontProperties(fname=font_path, size=16)
    font_prop_label = font_manager.FontProperties(fname=font_path, size=14)
    font_prop_tick = font_manager.FontProperties(fname=font_path, size=11)
    print(f"フォント '{font_path}' の読み込みに成功しました。")
except IOError:
    print(f"警告: 日本語フォントパス '{font_path}' が見つかりません。")
    font_prop_title, font_prop_subtitle, font_prop_label, font_prop_tick = [None] * 4
# -------------------------

try:
    # --- 全てのCSVファイルを読み込んで結合 ---
    csv_files = glob.glob('*_grid_search_behavior_experiment.csv')
    if not csv_files:
        raise FileNotFoundError("探索挙動の実験結果CSVファイルが見つかりません。")
    
    df_list = []
    for f in csv_files:
        df_list.append(pd.read_csv(f, dtype={'Param_ID': str}))
    df = pd.concat(df_list, ignore_index=True)
    print(f"{len(df_list)}個のCSVファイルを読み込み、結合しました。")

    # --- データの準備 ---
    df['is_feasible'] = df['PSO_Bottleneck'] >= 0
    df_valid_pso = df[df['is_feasible']].copy()
    if df_valid_pso.empty:
        raise ValueError("有効な解（実行可能解）が一つも見つかりませんでした。分析を中断します。")
        
    df_valid_pso.loc[:, 'Optimality'] = df_valid_pso.apply(
        lambda row: (row['PSO_Bottleneck'] / row['Optimal_Bottleneck']) * 100
        if row['Optimal_Bottleneck'] > 0 else 0,
        axis=1
    )
    
    graph_type_labels = {'random': 'ランダム', 'ba': 'BA'}
    df_valid_pso['graph_type_jp'] = df_valid_pso['graph_type'].map(graph_type_labels).fillna('不明')

    # --- グラフ描画 ---
    sns.set_theme(style="whitegrid", context="talk")

    # ★★★ 遅延倍率ごとにループして、個別のグラフを生成 ★★★
    unique_delays = sorted(df_valid_pso['delay_multiplier'].unique())

    for delay in unique_delays:
        
        df_subset = df_valid_pso[df_valid_pso['delay_multiplier'] == delay]
        
        if df_subset.empty: 
            print(f"スキップ: 遅延倍率={delay} の組み合わせには有効なデータがありません。")
            continue

        g = sns.catplot(
            data=df_subset,
            kind='box',
            x='Param_ID',
            y='Optimality',
            col='Num_Nodes',
            row='graph_type_jp',
            sharex=True,
            sharey=True,
            height=5,
            aspect=1.5,
            margin_titles=True,
            order=sorted(df_subset['Param_ID'].unique()) # A,B,C,D,Eの順に並べる
        )
        
        main_title = f'探索挙動(w, c1, c2)の比較 (遅延倍率 = {delay})'
        g.fig.suptitle(main_title, y=1.03, fontproperties=font_prop_title)
        g.set_axis_labels("パラメータ戦略ID", "最適度（%）", fontproperties=font_prop_label)
        g.set_titles(row_template="{row_name}", col_template="ノード数 = {col_name}", fontproperties=font_prop_subtitle)
        g.set(ylim=(0, 110))
        
        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')
                label.set_fontproperties(font_prop_tick)

        filename = f'plot_behavior_analysis_D{str(delay).replace(".", "")}.png'
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(filename)
        print(f"グラフ '{filename}' を保存しました。")
        plt.close(g.fig)

except FileNotFoundError as e:
    print(f"エラー: {e}")
except Exception as e:
    print(f"エラーが発生しました: {e}")