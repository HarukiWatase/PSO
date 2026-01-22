import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# --- 日本語フォント設定 ---
# ご自身の環境に合わせて、前回成功したフォントのパスを指定してください。
font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc' 
try:
    jp_font_prop = font_manager.FontProperties(fname=font_path)
    print(f"フォント '{font_path}' の読み込みに成功しました。")
except IOError:
    print(f"エラー: フォントパスが見つかりません。")
    jp_font_prop = None
# -------------------------

if jp_font_prop:
    try:
        # --- ファイルの読み込み ---
        # ★★★ ご自身の最新のCSVファイル名に書き換えてください ★★★
        file_path = '20250717_152426_pso_comparison.csv' # 例
        df = pd.read_csv(file_path)
        print(f"'{file_path}' の読み込みに成功しました。")

        # --- データの準備 ---
        # PSOが有効な解を見つけたかどうかのフラグを作成 (ボトルネック > 1.0)
        df['is_feasible'] = df['PSO_Bottleneck'] > 1.0
        
        # 最適度（最適解に対する達成率 %）を計算
        df_valid_pso = df[df['is_feasible']].copy()
        df_valid_pso.loc[:, 'Optimality'] = df_valid_pso.apply(
            lambda row: (row['PSO_Bottleneck'] / row['Optimal_Bottleneck']) * 100
            if row['Optimal_Bottleneck'] > 0 else 0,
            axis=1
        )

        # --- グラフ描画 ---
        sns.set_theme(style="whitegrid")

        # グラフ1: 実行可能解 発見率の比較
        # 条件ごとに、実行可能解を見つけられた割合（%）を計算
        feasibility_rate = df.groupby(['graph_type', 'delay_multiplier', 'pso_version'])['is_feasible'].mean().reset_index()
        feasibility_rate['is_feasible'] *= 100 # パーセンテージに変換

        plt.figure(figsize=(14, 7))
        ax1 = sns.barplot(x='graph_type', y='is_feasible', hue='pso_version', data=feasibility_rate, 
                          order=['random', 'grid', 'ba'], hue_order=['death_penalty', 'penalty_method'])
        ax1.set_title('PSOのバージョン別 実行可能解の発見率', fontproperties=jp_font_prop, fontsize=18)
        ax1.set_xlabel('グラフのトポロジー', fontproperties=jp_font_prop, fontsize=14)
        ax1.set_ylabel('実行可能解 発見率 (%)', fontproperties=jp_font_prop, fontsize=14)
        ax1.set_ylim(0, 110)
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles=handles, labels=['死のペナルティ', 'ペナルティ法'], title='PSOバージョン', prop=jp_font_prop, title_fontproperties=jp_font_prop)
        plt.savefig('plot_feasibility_rate.png')
        print("グラフ1 'plot_feasibility_rate.png' を保存しました。")


        # グラフ2: 解の精度（最適度）の比較
        plt.figure(figsize=(14, 7))
        ax2 = sns.boxplot(x='graph_type', y='Optimality', hue='pso_version', data=df_valid_pso, 
                          order=['random', 'grid', 'ba'], hue_order=['death_penalty', 'penalty_method'])
        ax2.set_title('PSOのバージョン別 解の精度（実行可能解のみ）', fontproperties=jp_font_prop, fontsize=18)
        ax2.set_xlabel('グラフのトポロジー', fontproperties=jp_font_prop, fontsize=14)
        ax2.set_ylabel('最適度 (%)', fontproperties=jp_font_prop, fontsize=14)
        ax2.set_ylim(0, 110)
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles=handles, labels=['死のペナルティ', 'ペナルティ法'], title='PSOバージョン', prop=jp_font_prop, title_fontproperties=jp_font_prop)
        plt.savefig('plot_optimality_comparison.png')
        print("グラフ2 'plot_optimality_comparison.png' を保存しました。")

    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")