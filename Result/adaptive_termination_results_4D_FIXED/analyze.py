# --- 設定 ---
# 読み込むCSVファイル名 (シミュレーションで生成されたファイル名に書き換えてください)
# CSV_FILENAME = '20251106_162401_4criteria_progress_log.csv' 


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import japanize_matplotlib # 日本語化

# --- 設定 ---
# 読み込むCSVファイル名 (シミュレーションで生成されたファイル名に書き換えてください)
CSV_FILENAME = '20251106_162401_4criteria_progress_log.csv' 

# グラフ化するノード数（シミュレーションで設定したもの）
NODE_TO_PLOT = 100

# PSOのタイムアウト時間（秒）。watase_24.py の設定 と合わせる
PSO_TIME_LIMIT = 1200.0 

# ボトルネックの最大値（グラフの上限）
MAX_BOTTLENECK = 20

# 出力する画像ファイル名
OUTPUT_IMAGE_NAME = f'average_progress_plot_N{NODE_TO_PLOT}_v5_convergence.png'
# --- 設定ここまで ---

def plot_average_time_vs_bottleneck(csv_path, target_node, max_bn, timeout_sec):
    """
    [v5 新ロジック]
    CSVを読み込み、「特定のボトルネック値に到達するまでの平均時間」を
    計算して、右肩上がりの収束曲線を描画する。
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"エラー: ファイル '{csv_path}' が見つかりません。")
        return

    # 対象のノード数のデータのみを抽出
    df_plot = df[df['Num_Nodes'] == target_node]
    if df_plot.empty:
        print(f"エラー: N={target_node} のデータがCSV内に見つかりません。")
        return

    # --- データを 'Exact' と 'PSO' に分離 ---
    df_pso = df_plot[df_plot['Algorithm'] == 'PSO']
    df_exact = df_plot[df_plot['Algorithm'] == 'Exact']

    # --- 描画処理 ---
    plt.figure(figsize=(12, 7))
    
    # 1. ★★★ 変更点: PSOの「平均収束曲線」を描画 ★★★
    if not df_pso.empty:
        print("PSOの平均収束曲線を計算中...")
        
        # ボトルネックの目標値 (1 から max_bn まで)
        target_bns = np.arange(1, max_bn + 1)
        
        # 各試行(Iter)ごとに、各目標BNを達成した「最短時間」を計算
        iter_groups = df_pso.groupby('Iter')
        
        # (目標BN, Iter) ごとの達成時間を格納するDataFrame
        time_to_reach_bn = pd.DataFrame(
            index=df_pso['Iter'].unique(), 
            columns=target_bns,
            dtype=float
        )

        for iter_id, group_df in iter_groups:
            for bn_target in target_bns:
                # この試行で、目標BN以上を達成したレコードを検索
                reached_records = group_df[group_df['Bottleneck'] >= bn_target]
                if not reached_records.empty:
                    # 達成した場合、その最短時間（最初の発見時間）を記録
                    min_time = reached_records['Time'].min()
                    time_to_reach_bn.loc[iter_id, bn_target] = min_time
        
        # ★重要★ 
        # 達成できなかった試行（NaN）をタイムアウト時間で埋める
        time_to_reach_bn = time_to_reach_bn.fillna(timeout_sec)
        
        # 各目標BNごとの「平均達成時間」を計算
        avg_time_per_bn = time_to_reach_bn.mean(axis=0)
        
        # (平均時間, ボトルネック) で折れ線グラフを描画
        plt.plot(
            avg_time_per_bn.values, # X軸: 平均到達時間
            avg_time_per_bn.index,  # Y軸: ボトルネック
            marker='o',
            linestyle='-',
            label='PSO (平均収束曲線)',
            color='tab:blue'
        )
        print("PSOの計算完了。")

    # 2. 厳密解法の平均点と横線を計算して描画 (v3と同様)
    if not df_exact.empty:
        exact_avg_time = df_exact['Time'].mean()
        exact_avg_bn = df_exact['Bottleneck'].mean()
        exact_label = f'厳密解法 (平均: {exact_avg_time:.2f}秒, BN: {exact_avg_bn:.1f})'
        
        plt.scatter(
            [exact_avg_time], [exact_avg_bn], 
            marker='x', color='red', s=150, zorder=10, label=exact_label 
        )
        xmin, xmax = plt.gca().get_xlim()
        # X軸の最大値を、厳密解法の時間 もしくは タイムアウト時間 の大きい方にする
        plot_xmax = max(xmax, exact_avg_time, timeout_sec * 0.1) 
        plt.plot(
            [exact_avg_time, plot_xmax], [exact_avg_bn, exact_avg_bn],
            color='red', linestyle='--'
        )
        plt.xlim(xmin, plot_xmax) # 描画範囲をリセット
        
        print(f"厳密解法の集計結果: 平均時間={exact_avg_time:.2f}秒, 平均BN={exact_avg_bn:.1f}")

    # --- グラフの体裁 ---
    plt.xlabel('計算時間 (秒)', fontsize=14)
    plt.ylabel('ボトルネック帯域', fontsize=14)
    graph_type = df_plot['Graph_Type'].iloc[0]
    num_iter = df_plot['Iter'].nunique()
    
    plt.title(f'計算時間とボトルネック帯域の比較 (N={target_node}, {graph_type}, {num_iter}回試行)', fontsize=16)

    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()

    # グラフを画像ファイルとして保存
    plt.savefig(OUTPUT_IMAGE_NAME)
    print(f"平均グラフ(v5 収束曲線)を '{OUTPUT_IMAGE_NAME}' として保存しました。")

if __name__ == '__main__':
    plot_average_time_vs_bottleneck(CSV_FILENAME, NODE_TO_PLOT, MAX_BOTTLENECK, PSO_TIME_LIMIT)