# analyze_time_comparison_final_v6.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

def create_final_style_plot(csv_filepath):
    """
    テキスト要素ごとに日本語フォントを明示的に指定し、文字化けを完全に解消する。
    """
    # --- 1. 日本語フォントの読み込み ---
    font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc' 
    try:
        jp_font_prop = font_manager.FontProperties(fname=font_path)
        print(f"日本語フォント '{font_path}' の読み込みに成功しました。")
    except IOError:
        print(f"警告: 日本語フォント '{font_path}' が見つかりません。")
        jp_font_prop = font_manager.FontProperties(family='sans-serif')

    # --- 2. ファイルの読み込み ---
    try:
        df = pd.read_csv(csv_filepath)
        print(f"'{csv_filepath}' の読み込みに成功しました。")
    except FileNotFoundError:
        print(f"エラー: ファイル '{csv_filepath}' が見つかりません。")
        return

    # --- 3. データの準備 ---
    df_time = df.rename(columns={'Optimal_Time': '厳密解法', 'PSO_Time': 'PSO'})
    graph_type_labels = {'random': 'ランダム', 'grid': 'グリッド', 'ba': 'BAモデル'}
    df_time['graph_type_jp'] = df_time['Graph_Type'].map(graph_type_labels)

    df_time_melted = df_time.melt(
        id_vars=['Num_Nodes', 'graph_type_jp'], 
        value_vars=['厳密解法', 'PSO'],
        var_name='アルゴリズム',
        value_name='計算時間 (秒)'
    )

    # --- 4. グラフ描画 ---
    sns.set_theme(style="ticks", context="talk")

    g = sns.relplot(
        data=df_time_melted,
        kind='line',
        x='Num_Nodes',
        y='計算時間 (秒)',
        hue='アルゴリズム',
        style='アルゴリズム',
        col='graph_type_jp',
        markers={'厳密解法': 'o', 'PSO': '^'},
        dashes=False,
        lw=2.5,
        palette={'PSO': 'red', '厳密解法': 'blue'},
        height=5,
        aspect=1.1,
        ci=None
    )

    # ★★★ 修正点1: 各テキスト要素に fontproperties を直接指定 ★★★
    g.fig.suptitle('ネットワーク規模と計算時間の関係（トポロジー別）', y=1.05, fontproperties=jp_font_prop, fontsize=20)
    g.set_axis_labels("ノード数", "平均計算時間（秒）", fontproperties=jp_font_prop, fontsize=14)
    g.set_titles("{col_name}", fontproperties=jp_font_prop, fontsize=16)
    g.set(yscale="log")
    
    # 枠線と目盛りのスタイルを設定
    for ax in g.axes.flat:
        for spine_pos in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine_pos].set_visible(True)
            ax.spines[spine_pos].set_linewidth(0.8)
        ax.tick_params(width=0.8, length=5)

    # ★★★ 修正点2: 凡例のテキストにも fontproperties を確実に適用 ★★★
    legend = g.legend
    legend.set_title("アルゴリズム", prop=jp_font_prop)
    for text in legend.get_texts():
        text.set_fontproperties(jp_font_prop)

    # --- 6. 保存 ---
    output_filename = 'plot_time_comparison_final_style.png'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"グラフ '{output_filename}' を保存しました。")


if __name__ == '__main__':
    # ★★★ ここに分析したいCSVファイルの名前を入力してください ★★★
    target_csv = '20251020_052840_adaptive_termination.csv' # or your latest result file
    create_final_style_plot(target_csv)

