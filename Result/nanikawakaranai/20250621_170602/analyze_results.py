import pandas as pd
import numpy as np
import glob
import os

def analyze_pso_performance(file_path):
    """
    指定されたCSVファイルを読み込み、PSOの性能評価（最適解発見率、近似率）を行う。
    """
    print(f"\n📄 分析対象ファイル: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"   [エラー] ファイルが見つかりません。")
        return None

    # --- 評価指標の計算 ---

    # カラム名を短縮して扱いやすくする
    optimal_bn_col = 'Constrained_Mod_Dijkstra_BN (Optimal)'
    pso_bn_col = 'PSO_Bottleneck'

    # 1. 最適解発見フラグ (Is_Optimal)
    # PSOの解と最適解が一致した行にTrueを設定
    df['Is_Optimal'] = (df[pso_bn_col] == df[optimal_bn_col])

    # 2. 近似率 (Approximation_Ratio)
    # 最適解が0より大きい場合のみ計算（0除算を避ける）
    # 最適解が0の場合は、近似率をNaN（非数）として扱う
    df['Approximation_Ratio'] = np.where(
        df[optimal_bn_col] > 0,
        df[pso_bn_col] / df[optimal_bn_col],
        np.nan
    )

    # --- 遅延倍率ごとに集計 ---
    
    # 'delay_multiplier' でグループ化
    grouped = df.groupby('delay_multiplier')
    
    # 集計結果を格納するリスト
    analysis_results = []

    for multiplier, group in grouped:
        
        # 近似率の計算（NaN値は除外して計算）
        valid_approximations = group['Approximation_Ratio'].dropna()
        
        if not valid_approximations.empty:
            avg_approx_ratio = valid_approximations.mean()
            std_approx_ratio = valid_approximations.std()
            min_approx_ratio = valid_approximations.min()
            max_approx_ratio = valid_approximations.max()
        else:
            avg_approx_ratio, std_approx_ratio, min_approx_ratio, max_approx_ratio = np.nan, np.nan, np.nan, np.nan
        
        # 最適解発見率の計算
        total_trials = len(group)
        optimality_rate = group['Is_Optimal'].sum() / total_trials if total_trials > 0 else 0
        
        analysis_results.append({
            'Delay_Multiplier': multiplier,
            'Optimality_Rate': optimality_rate,
            'Avg_Approximation_Ratio': avg_approx_ratio,
            'Std_Dev_Approximation': std_approx_ratio,
            'Min_Approximation_Ratio': min_approx_ratio,
            'Max_Approximation_Ratio': max_approx_ratio,
            'Total_Trials': total_trials,
        })

    # 結果をデータフレームに変換して返す
    return pd.DataFrame(analysis_results)


if __name__ == '__main__':
    # 'results' ディレクトリ内の最新のCSVファイルを探す（ファイルパスは環境に合わせて変更してください）
    # この例では、カレントディレクトリにあるCSVを対象とします。
    # result_path = 'results/*.csv' のようにディレクトリを指定することも可能です。
    list_of_files = glob.glob('*.csv') 
    
    if not list_of_files:
        print("分析対象のCSVファイルが見つかりません。")
    else:
        # 最新のファイルを選択
        latest_file = max(list_of_files, key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0)
        
        # 分析を実行
        summary_df = analyze_pso_performance(latest_file)

        if summary_df is not None:
            print("\n===== PSO性能評価サマリー =====")
            # to_string() を使うとターミナルできれいに表示される
            print(summary_df.to_string())

            # (オプション) サマリー結果を新しいCSVファイルに保存
            summary_filename = 'analysis_summary.csv'
            summary_df.to_csv(summary_filename, index=False)
            print(f"\n✅ 評価サマリーを '{summary_filename}' に保存しました。")