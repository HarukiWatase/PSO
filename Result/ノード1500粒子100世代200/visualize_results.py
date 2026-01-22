import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# --- 日本語フォント設定（ファイルパス直接指定） ---
# ご自身のPCで確認いただいた、フォントファイルへの完全なパスを設定してください。
font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc' 

# FontPropertiesオブジェクトを作成
try:
    jp_font_prop = font_manager.FontProperties(fname=font_path)
    print(f"フォント '{font_path}' の読み込みに成功しました。")
except IOError:
    print(f"エラー: 指定されたパスにフォントファイルが見つかりません。")
    print("「Font Book.app」でパスを確認し、正しいパスに書き換えてください。")
    jp_font_prop = None
# -------------------------------------------------------------

if jp_font_prop:
    try:
        # --- ファイルの読み込み ---
        file_path = '20250713_005230_with_relative_delay.csv'
        df = pd.read_csv(file_path)
        print(f"'{file_path}' のCSV読み込みに成功しました。")

        # --- データの準備 ---
        df_valid_pso = df[df['PSO_Bottleneck'] > 1.0].copy()
        df_valid_pso.loc[:, 'Optimality'] = df_valid_pso.apply(
            lambda row: (row['PSO_Bottleneck'] / row['Label_Correcting_BN (Optimal)']) * 100
            if row['Label_Correcting_BN (Optimal)'] > 0 else 0,
            axis=1
        )

        # --- グラフ描画 ---
        sns.set_theme(style="whitegrid")

        # グラフ1: 解の精度（最適度）の比較
        plt.figure(figsize=(10, 6))
        ax1 = sns.boxplot(x='delay_multiplier', y='Optimality', data=df_valid_pso)
        ax1.set_title('遅延制約の厳しさ別 PSOの解の精度', fontproperties=jp_font_prop, fontsize=16)
        ax1.set_xlabel('遅延倍率（値が小さいほど制約が厳しい）', fontproperties=jp_font_prop, fontsize=12)
        ax1.set_ylabel('最適解(ラベル訂正)に対する精度(%)', fontproperties=jp_font_prop, fontsize=12)
        plt.ylim(0, 110)
        plt.savefig('plot_optimality_jp.png')
        print("グラフ1 'plot_optimality_jp.png' を保存しました。")

        # グラフ2: 平均計算時間の比較
        time_avg = df.groupby('delay_multiplier')[['PSO_Time_sec', 'Label_Correcting_Time_sec']].mean().reset_index()
        time_melted = time_avg.melt(id_vars='delay_multiplier', var_name='Algorithm', value_name='Time (sec)')
        time_melted['Algorithm'] = time_melted['Algorithm'].replace({
            'PSO_Time_sec': 'PSO',
            'Label_Correcting_Time_sec': 'ラベル訂正アルゴリズム'
        })

        plt.figure(figsize=(10, 6))
        ax2 = sns.barplot(x='delay_multiplier', y='Time (sec)', hue='Algorithm', data=time_melted)
        ax2.set_title('アルゴリズム別 平均計算時間の比較', fontproperties=jp_font_prop, fontsize=16)
        ax2.set_xlabel('遅延倍率', fontproperties=jp_font_prop, fontsize=12)
        ax2.set_ylabel('平均計算時間（秒）', fontproperties=jp_font_prop, fontsize=12)
        ax2.set_yscale('log')
        
        # ★★★ 修正点 ★★★
        # 凡例のタイトルにもフォントプロパティを適用
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles=handles, labels=labels, title='アルゴリズム', 
                   prop=jp_font_prop, title_fontproperties=jp_font_prop)
        plt.savefig('plot_average_time_jp.png')
        print("グラフ2 'plot_average_time_jp.png' を保存しました。")

        # グラフ3: 計算時間の分布（散布図）
        plt.figure(figsize=(9, 8))
        ax3 = sns.scatterplot(x='Label_Correcting_Time_sec', y='PSO_Time_sec', data=df, hue='delay_multiplier', palette='viridis', s=60, alpha=0.8)
        max_time_val = df[['PSO_Time_sec', 'Label_Correcting_Time_sec']].max().max()
        if pd.notnull(max_time_val):
            ax3.plot([0, max_time_val], [0, max_time_val], color='red', linestyle='--', label='y = x (同じ計算時間)')

        ax3.set_title('計算時間の分布比較（PSO vs ラベル訂正法）', fontproperties=jp_font_prop, fontsize=16)
        ax3.set_xlabel('ラベル訂正アルゴリズムの計算時間（秒）', fontproperties=jp_font_prop, fontsize=12)
        ax3.set_ylabel('PSOの計算時間（秒）', fontproperties=jp_font_prop, fontsize=12)

        # ★★★ 修正点 ★★★
        # 凡例のタイトルと項目にフォントプロパティを適用
        handles, labels = ax3.get_legend_handles_labels()
        ax3.legend(handles=handles, labels=labels, title='凡例', 
                   prop=jp_font_prop, title_fontproperties=jp_font_prop)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('plot_time_distribution_jp.png')
        print("グラフ3 'plot_time_distribution_jp.png' を保存しました。")

    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません。")
        print("このスクリプトをCSVファイルと同じディレクトリに置いて実行してください。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")