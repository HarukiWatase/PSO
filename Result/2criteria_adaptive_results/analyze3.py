# analyze_results_separated_jp_fix5.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib # ★ 1. japanize_matplotlib をインポート

# ★ 2. 凡例などでマイナス記号が文字化けするのを防ぎます
plt.rcParams['axes.unicode_minus'] = False 

def analyze_and_plot_separately(csv_filepath):
    """
    シミュレーション結果を読み込み、グラフの種類ごとに計算時間の比較グラフを生成する。
    (ポスター発表用に日本語化・スタイル修正済み)
    """
    # --- 1. データの読み込み ---
    try:
        df = pd.read_csv(csv_filepath)
        print(f"'{csv_filepath}' を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイル '{csv_filepath}' が見つかりません。ファイル名とパスを確認してください。")
        return
    except Exception as e:
        print(f"ファイルの読み込み中に予期せぬエラーが発生しました: {e}")
        return

    # --- 2. データの前処理 ---
    time_cols = ['Optimal_Time', 'PSO_Time']
    for col in time_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    graph_types = df['Graph_Type'].unique()
    print(f"ファイルに含まれるグラフの種類: {graph_types}")

    # --- 3. グラフ種類ごとにループして描画 ---
    for g_type in graph_types:
        print(f"\n--- グラフ '{g_type}' のデータを処理中... ---")
        
        df_filtered = df[df['Graph_Type'] == g_type].copy()

        if df_filtered.empty:
            print(f"'{g_type}' のデータが見つかりません。スキップします。")
            continue

        results_agg = df_filtered.groupby('Num_Nodes')[time_cols].mean().reset_index()
        
        print("集計結果:")
        print(results_agg)

        # (sns.set_style() はフォントリセットを防ぐため呼び出さない)
        plt.figure(figsize=(10, 6))

        # 厳密解法の結果をプロット
        plt.plot(results_agg['Num_Nodes'], results_agg['Optimal_Time'], 
                 marker='o', linestyle='-', color='blue', label='厳密解法') # 凡例を日本語化

        # PSOのスタイルを (三角・実線) に変更
        plt.plot(results_agg['Num_Nodes'], results_agg['PSO_Time'], 
                 marker='^',      # マーカーを三角に変更
                 linestyle='-',     # 線種を実線に変更
                 color='red', 
                 label='PSO')   # 凡例を日本語化

        # グラフの体裁設定を日本語化
        plt.title(f'1制約問題における平均計算時間の推移', fontsize=28)
        plt.xlabel('ノード数', fontsize=24)
        plt.ylabel('平均計算時間 (秒)', fontsize=24)
        
        plt.yscale('log') # 縦軸の対数スケールは維持
        plt.legend(fontsize=12)
        
        # グリッドを明示的にオフにする (ポスター用)
        plt.grid(False)
        
        # ★★★ 変更点 ★★★
        # sns.despine() を削除し、デフォルトの四方枠線に戻す
        # sns.despine() 
        # ★★★ 変更ここまで ★★★
        plt.ylim(bottom=0.002, top=1200)
        # グラフを種類ごとに個別のファイルとして保存
        output_filename = f'計算時間比較_{g_type}.png' # ファイル名を日本語化
        plt.savefig(output_filename)
        print(f"グラフを '{output_filename}' として保存しました。")


if __name__ == '__main__':
    # ★★★ ここに分析したいCSVファイルの名前を入力してください ★★★
    target_csv = '20251102_150356_2criteria_adaptive.csv'
    
    analyze_and_plot_separately(target_csv)