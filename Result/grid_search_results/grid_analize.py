import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import glob

# --- 日本語フォント設定 ---
font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W0.ttc'
try:
    font_prop_title = font_manager.FontProperties(fname=font_path, size=22)
    font_prop_subtitle = font_manager.FontProperties(fname=font_path, size=16)
    font_prop_label = font_manager.FontProperties(fname=font_path, size=14)
    font_prop_tick = font_manager.FontProperties(fname=font_path, size=11)
    print(f"フォント '{font_path}' の読み込みに成功しました。")
except IOError:
    print(f"警告: 日本語フォントパス '{font_path}' が見つかりません。")
    font_prop_title, font_prop_subtitle, font_prop_label, font_prop_tick = [None] * 4
# -------------------------

try:
    # --- 全てのCSVファイルを読み込んで結合 ---
    csv_files = glob.glob('*_grid_search_experiment.csv')
    if not csv_files:
        raise FileNotFoundError("グリッドサーチのCSVファイルが見つかりません。")
    
    print("--- デバッグ情報 ---")
    print(f"発見したCSVファイル ({len(csv_files)}個):")
    for f in csv_files:
        print(f"  - {f}")
    
    df_list = []
    for f in csv_files:
        df_list.append(pd.read_csv(f, dtype={'Param_ID': str}))
    df = pd.concat(df_list, ignore_index=True)
    print(f"\n全CSVファイルの結合に成功しました。総行数: {len(df)}")

    # --- データの準備 ---
    df['is_feasible'] = df['PSO_Bottleneck'] >= 0
    
    def get_axis(param_id):
        if 'Cost' in param_id: return '1. 探索コスト'
        elif 'Behav' in param_id: return '2. 探索挙動'
        elif 'Pen' in param_id: return '3. ペナルティ強度'
        return 'Unknown'
    df['Axis'] = df['Param_ID'].apply(get_axis)
    
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★ ここからが追加した診断機能 ★★★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    print("\n--- 実行可能解の発見率サマリー ---")
    # 調査軸ごとに、実行可能解が見つかった(True)か、見つからなかった(False)かの割合を計算
    summary = df.groupby('Axis')['is_feasible'].value_counts(normalize=True).unstack(fill_value=0) * 100
    print("各調査軸で、実行可能解が見つかった試行の割合（%）:")
    print(summary)
    print("------------------------------------\n")
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    
    # グラフ描画のために、実行可能解が見つかったデータのみを抽出
    df_valid_pso = df[df['is_feasible']].copy()
    if df_valid_pso.empty:
        print("警告: 全ての試行で実行可能解が見つからなかったため、グラフは生成されません。")
    else:
        df_valid_pso.loc[:, 'Optimality'] = df_valid_pso.apply(
            lambda row: (row['PSO_Bottleneck'] / row['Optimal_Bottleneck']) * 100
            if row['Optimal_Bottleneck'] > 0 else 0,
            axis=1
        )

        graph_type_labels = {'random': 'ランダム', 'ba': 'BAモデル'}
        df_valid_pso['graph_type_jp'] = df_valid_pso['graph_type'].map(graph_type_labels).fillna('不明')

        # --- グラフ描画 ---
        sns.set_theme(style="whitegrid", context="talk")

        unique_nodes = sorted(df_valid_pso['Num_Nodes'].unique())
        unique_graphs = sorted(df_valid_pso['graph_type_jp'].unique())
        unique_delays = sorted(df_valid_pso['delay_multiplier'].unique())

        for nodes in unique_nodes:
            for graph in unique_graphs:
                for delay in unique_delays:
                    
                    df_subset = df_valid_pso[
                        (df_valid_pso['Num_Nodes'] == nodes) &
                        (df_valid_pso['graph_type_jp'] == graph) &
                        (df_valid_pso['delay_multiplier'] == delay)
                    ]
                    
                    if df_subset.empty: 
                        print(f"スキップ: N={nodes}, G={graph}, D={delay} の組み合わせには有効なデータがありません。")
                        continue

                    fig, axes = plt.subplots(1, 3, figsize=(28, 8))
                    fig.suptitle(f'パラメータ比較: {nodes}ノード, {graph}, 遅延倍率={delay}', y=1.03, fontproperties=font_prop_title)

                    axes_map = {
                        '1. 探索コスト': axes[0],
                        '2. 探索挙動': axes[1],
                        '3. ペナルティ強度': axes[2],
                    }
                    
                    all_axes_empty = True
                    for axis_name, ax in axes_map.items():
                        ax_data = df_subset[df_subset['Axis'] == axis_name]
                        if not ax_data.empty:
                            all_axes_empty = False
                            sns.boxplot(data=ax_data, x='Param_ID', y='Optimality', ax=ax)
                            ax.set_title(axis_name, fontproperties=font_prop_subtitle)
                            ax.set_ylim(0, 110)
                            ax.set_xlabel("パラメータセットID", fontproperties=font_prop_label)
                            ax.set_ylabel("最適度（%）", fontproperties=font_prop_label)
                            for label in ax.get_xticklabels():
                                label.set_rotation(30)
                                label.set_ha('right')
                                label.set_fontproperties(font_prop_tick)
                        else:
                            # データがない場合の表示
                            ax.text(0.5, 0.5, '有効なデータなし', ha='center', va='center', fontproperties=font_prop_label)
                            ax.set_title(axis_name, fontproperties=font_prop_subtitle)


                    if not all_axes_empty:
                        graph_name_short = ''.join(filter(str.isalnum, graph))
                        filename = f'plot_N{nodes}_G{graph_name_short}_D{str(delay).replace(".", "")}.png'
                        
                        plt.tight_layout(rect=[0, 0, 1, 0.96])
                        plt.savefig(filename)
                        print(f"グラフ '{filename}' を保存しました。")
                    plt.close(fig)

except FileNotFoundError as e:
    print(f"エラー: {e}")
except Exception as e:
    print(f"エラーが発生しました: {e}")