import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # --- ファイルの読み込み ---
    # このスクリプトをCSVファイルと同じフォルダに置いて実行してください
    file_path = '20250713_005230_with_relative_delay.csv'
    df = pd.read_csv(file_path)
    print(f"'{file_path}' の読み込みに成功しました。")

    # --- グラフ描画 ---
    sns.set_theme(style="whitegrid")

    # delay_multiplier の値ごとにループしてグラフを生成
    for multiplier in df['delay_multiplier'].unique():
        
        # 現在のmultiplierの値でデータを絞り込み
        df_filtered = df[df['delay_multiplier'] == multiplier]

        # グラフの描画領域を作成
        plt.figure(figsize=(14, 7))

        # 1. PSOの実行時間をプロット
        sns.lineplot(data=df_filtered, x='iter', y='PSO_Time_sec', 
                     label='PSO Execution Time', color='royalblue', marker='o', markersize=5)
        
        # 2. ラベル訂正アルゴリズムの実行時間をプロット
        sns.lineplot(data=df_filtered, x='iter', y='Label_Correcting_Time_sec', 
                     label='Label-Correcting Execution Time', color='green', marker='o', markersize=5)

        # グラフのタイトルとラベルを設定（英語表記）
        plt.title(f'Execution Time per Trial (Delay Multiplier = {multiplier})', fontsize=16)
        plt.xlabel('Trial Number (Iteration)', fontsize=12)
        plt.ylabel('Execution Time (sec) - Log Scale', fontsize=12)
        
        # Y軸を対数スケールに設定（値の差が大きいため）
        plt.yscale('log')
        
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.xticks(range(1, len(df_filtered) + 1)) # X軸の目盛りを整数にする
        plt.tight_layout()

        # グラフをファイルとして保存
        save_path = f'plot_time_vs_trials_{str(multiplier).replace(".", "_")}.png'
        plt.savefig(save_path)
        print(f"グラフ '{save_path}' を保存しました。")

except FileNotFoundError:
    print(f"エラー: ファイル '{file_path}' が見つかりません。")
    print("このスクリプトをCSVファイルと同じディレクトリに置いて実行してください。")
except Exception as e:
    print(f"エラーが発生しました: {e}")