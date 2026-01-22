import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import os
# ★★★ 1. ファイル名とノード数の設定を編集してください ★★★
# 実行結果のCSVファイル名を指定
CSV_FILENAME = 'compare_all_20251207_191850.csv'  # <--- 適切なファイル名に置き換えてください
TARGET_N = 1000
# ★★★ --------------------------------------- ★★★


def calculate_and_plot_ratio_no_errorbar(filename, target_n):
    if not os.path.exists(filename):
        print(f"エラー: ファイル '{filename}' が見つかりません。")
        return

    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"ファイル読み込みエラー: {e}")
        return

    # 1. データのフィルタリング
    df_target = df[df['NodeCount'] == target_n].copy()
    
    # 厳密解(Exact)が有効なデータのみ抽出
    df_valid = df_target[df_target['Exact_BN'] > 0]
    
    if df_valid.empty:
        print(f"エラー: ノード数 {target_n} の有効なデータがありません。")
        return

    # 2. 近似率の計算
    df_valid['Global_Ratio'] = df_valid['Global_BN'] / df_valid['Exact_BN'] * 100
    df_valid['Spatial_Ratio'] = df_valid['Spatial_BN'] / df_valid['Exact_BN'] * 100
    df_valid['Restart_Ratio'] = df_valid['Restart_BN'] / df_valid['Exact_BN'] * 100

    # 3. 集計
    data_for_plot = {
        'Global PSO': df_valid['Global_Ratio'],
        'Local PSO': df_valid['Spatial_Ratio'],
        'Restart PSO': df_valid['Restart_Ratio']
    }
    
    df_ratio = pd.DataFrame(data_for_plot)
    mean_ratios = df_ratio.mean()
    
    print(f"\n--- N={target_n} 集計結果 (n={len(df_valid)}) ---")
    print(mean_ratios)

    # 4. グラフ作成
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # ★変更点: yerr引数を削除しました
    bars = ax.bar(mean_ratios.index, mean_ratios.values, 
                  color=['#aec7e8', '#ffbb78', '#2ca02c'], 
                  edgecolor='black', linewidth=1)
    
    ax.tick_params(axis='x', labelsize=16)

    # ★変更点: Y軸の上限を平均値ベースで調整
    max_height = mean_ratios.max()
    upper_limit = 100
    ax.set_ylim(0, upper_limit)
    
    # Optimalライン

    
    ax.set_ylabel('解精度(%)', fontsize=16)
    ax.legend(loc='upper left')
    
    # 値の表示
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, 
                f'{yval:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    calculate_and_plot_ratio_no_errorbar(CSV_FILENAME, TARGET_N)