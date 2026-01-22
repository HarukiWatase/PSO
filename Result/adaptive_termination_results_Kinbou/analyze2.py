import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import japanize_matplotlib # 日本語化

# --- 設定 ---
# 読み込むCSVファイル名 (watase_26 が生成したファイル名に書き換えてください)
SUMMARY_CSV = '20251129_162030_kinbou_summary.csv'
GENERATION_LOG_CSV = '20251129_162030_kinbou_generation_log.csv'

# グラフ化するノード数（シミュレーションで設定したもの）
NODE_TO_PLOT = 1000

# ★★★ 変更点 (1/2) ★★★
# グラフに表示するX軸（世代）の最大値を設定
# (例: 100世代以降は不要な場合、150 や 200 に設定)
X_AXIS_LIMIT = 100 
# ★★★ 変更ここまで ★★★

# 出力する画像ファイル名
OUTPUT_IMAGE_NAME = f'par300_{NODE_TO_PLOT}_xlim{X_AXIS_LIMIT}.png'
# --- 設定ここまで ---

def plot_generation_vs_bottleneck(summary_csv, generation_csv, target_node, x_limit):
    """
    [v2 修正点]
    - X軸の表示範囲を (0 ～ x_limit) に制限
    - PSOグラフの信頼区間(ci)を非表示
    """
    try:
        df_summary = pd.read_csv(summary_csv)
    except FileNotFoundError:
        print(f"エラー: サマリーCSV '{summary_csv}' が見つかりません。")
        return
        
    try:
        df_gen_log = pd.read_csv(generation_csv)
    except FileNotFoundError:
        print(f"エラー: 世代ログCSV '{generation_csv}' が見つかりません。")
        return

    # 対象ノードのデータに絞る
    df_summary = df_summary[df_summary['Num_Nodes'] == target_node]
    df_gen_log = df_gen_log[df_gen_log['Num_Nodes'] == target_node]

    if df_gen_log.empty or df_summary.empty:
        print(f"エラー: N={target_node} のデータがCSV内に見つかりません。")
        return

    # --- 描画処理 ---
    plt.figure(figsize=(12, 7))

    # 1. ★★★ 変更点: PSOの「世代別平均」を計算し、「折れ線」で描画 ★★★
    df_gen_log['Bottleneck'] = df_gen_log['Bottleneck'].replace(-1, 0)
    
    print("PSOの世代別平均を計算中...")
    # 'Generation' ごとに 'Bottleneck' の平均値を計算
    avg_bn_per_gen = df_gen_log.groupby('Generation')['Bottleneck'].mean()
    
    # matplotlib の plot を使い、平均値を折れ線で描画
    plt.plot(
        avg_bn_per_gen.index,  # X軸: 世代
        avg_bn_per_gen.values, # Y軸: 平均ボトルネック
        label='PSO (世代別平均)',
        color='tab:blue',
        marker='.',        # 世代ごとに点を打つ (不要なら marker=None)
        markersize=4
    )
    # ★★★ 変更ここまで ★★★

    # 2. 厳密解法の「推測点」と「時間の壁（縦線）」を描画 (変更なし)
    exact_avg_time = df_summary['Optimal_Time'].mean()
    exact_avg_bn = df_summary['Optimal_BN'].mean()
    avg_time_per_gen = df_gen_log.groupby('Generation')['Time'].mean()
    inferred_gen = avg_time_per_gen[avg_time_per_gen >= exact_avg_time].index.min()
    
    if pd.isna(inferred_gen):
        inferred_gen = avg_time_per_gen.index.max()
        
    print(f"厳密解法の集計結果: 平均時間={exact_avg_time:.2f}秒, 平均BN={exact_avg_bn:.1f}")
    print(f"推測世代: 厳密解法の平均時間は、PSOの {inferred_gen} 世代目に相当します。")

    plt.axvline(
        x=inferred_gen, 
        color='red', 
        linestyle='--', 
        label=f'厳密解法の平均時間 ({exact_avg_time:.2f}秒)'
    )
    plt.scatter(
        [inferred_gen], 
        [exact_avg_bn], 
        marker='x', 
        color='red', 
        s=150, 
        zorder=10,
        label=f'厳密解法 (平均BN: {exact_avg_bn:.1f})'
    )

    # --- グラフの体裁 ---
    plt.xlabel('世代 (Generation)', fontsize=14)
    plt.ylabel('ボトルネック帯域', fontsize=14)
    graph_type = df_gen_log['Graph_Type'].iloc[0]
    num_iter = df_summary['Iter'].nunique()
    
    plt.title(f'世代数とボトルネック帯域の比較 (N={target_node}, {graph_type}, {num_iter}回平均)', fontsize=16)
    
    # ★★★ 変更点 (1/2) 適用 ★★★
    plt.xlim(left=0, right=x_limit)
    # ★★★ 変更ここまで ★★★
    
    # Y軸の下限を0に固定（-1の置き換えに対応）
    plt.ylim(bottom=0) 

    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # グラフを画像ファイルとして保存
    plt.savefig(OUTPUT_IMAGE_NAME)
    print(f"世代グラフ(v2)を '{OUTPUT_IMAGE_NAME}' として保存しました。")

if __name__ == '__main__':
    # 2つのCSVファイル名とノード数を設定してください
    plot_generation_vs_bottleneck(SUMMARY_CSV, GENERATION_LOG_CSV, NODE_TO_PLOT, X_AXIS_LIMIT)