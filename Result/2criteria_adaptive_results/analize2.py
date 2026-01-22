# analyze_results_separated.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_and_plot_separately(csv_filepath):
    """
    シミュレーション結果を読み込み、グラフの種類ごとに計算時間の比較グラフを生成する。
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
    # 時間の列を数値に変換
    time_cols = ['Optimal_Time', 'PSO_Time']
    for col in time_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # CSVファイルに含まれるグラフの種類をすべて取得
    graph_types = df['Graph_Type'].unique()
    print(f"ファイルに含まれるグラフの種類: {graph_types}")

    # --- 3. グラフ種類ごとにループして描画 ---
    for g_type in graph_types:
        print(f"\n--- グラフ '{g_type}' のデータを処理中... ---")
        
        # 現在のグラフ種類のデータのみを抽出
        df_filtered = df[df['Graph_Type'] == g_type].copy()

        if df_filtered.empty:
            print(f"'{g_type}' のデータが見つかりません。スキップします。")
            continue

        # ノード数ごとに平均計算時間を集計
        results_agg = df_filtered.groupby('Num_Nodes')[time_cols].mean().reset_index()
        
        print("集計結果:")
        print(results_agg)

        # 描画
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))

        # 厳密解法の結果をプロット
        plt.plot(results_agg['Num_Nodes'], results_agg['Optimal_Time'], 
                 marker='o', linestyle='-', color='blue', label='Exact Algorithm')

        # PSOの結果をプロット
        plt.plot(results_agg['Num_Nodes'], results_agg['PSO_Time'], 
                 marker='x', linestyle='--', color='red', label='PSO')

        # グラフの体裁設定
        plt.title(f'Computation Time vs. Network Size ({g_type.upper()} Graph)', fontsize=16)
        plt.xlabel('Number of Nodes', fontsize=12)
        plt.ylabel('Average Computation Time (seconds)', fontsize=12)
        plt.yscale('log')
        plt.legend(fontsize=12)
        plt.grid(True, which="both", ls="--")
        
        # グラフを種類ごとに個別のファイルとして保存
        output_filename = f'naew_computation_time_comparison_{g_type}.png'
        plt.savefig(output_filename)
        print(f"グラフを '{output_filename}' として保存しました。")


if __name__ == '__main__':
    # ★★★ ここに分析したいCSVファイルの名前を入力してください ★★★
    target_csv = '20251102_150356_2criteria_adaptive.csv'
    
    analyze_and_plot_separately(target_csv)