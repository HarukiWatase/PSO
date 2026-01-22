# watase_15.py

import numpy as np
import random
import networkx as nx
import create_graph_3
from datetime import datetime
import csv
import savef 
from heapq import heappop, heappush
import time
import math # ★★★ インポート追加 ★★★

"""
This script runs a 3-criteria label-correcting algorithm.
PSO and other algorithms are disabled.
"""

###########
### 関数 ###
###########

# (PathEncodeからpath2delayまでの関数群はPSOでのみ使用するため、今回は使われません)

# ★★★ 変更点：厳密解法を3基準（帯域、遅延、ロス率）に対応させる ★★★
def find_optimal_path_by_label_correcting_3d(G, source, target, max_delay, max_loss_rate, 
                                             weight="weight", delay="delay", loss_cost="loss_log_cost"):
    """
    3-criteria label-correcting algorithm.
    Finds the path with the maximum bottleneck that satisfies delay and loss constraints.
    """
    labels = {node: [] for node in G.nodes()}
    pred = {node: {} for node in G.nodes()}
    pq = []

    # ラベル: (delay, bottle, loss_log_cost)
    initial_delay = 0.0
    initial_bottle = float('inf')
    initial_loss_cost = 0.0
    start_label = (initial_delay, initial_bottle, initial_loss_cost)
    
    labels[source].append(start_label)
    pred[source][start_label] = (None, None)
    
    # 優先度付きキュー: (-bottle, delay, loss_cost, node, label)
    heappush(pq, (-initial_bottle, initial_delay, initial_loss_cost, source, start_label))

    while pq:
        neg_bottle, current_delay, current_loss_cost, u, current_label = heappop(pq)
        current_bottle = -neg_bottle

        if current_label not in labels[u]:
            continue

        for v in G.neighbors(u):
            edge_data = G.get_edge_data(u, v)
            edge_weight = edge_data.get(weight, 1)
            edge_delay = edge_data.get(delay, 0)
            edge_loss_cost = edge_data.get(loss_cost, 0)

            new_delay = current_delay + edge_delay
            new_bottle = min(current_bottle, edge_weight)
            new_loss_cost = current_loss_cost + edge_loss_cost
            
            # 制約チェック
            if new_delay > max_delay:
                continue
            # 対数コストから実際のロス率を計算してチェック
            current_path_loss_rate = 1 - math.exp(-new_loss_cost)
            if current_path_loss_rate > max_loss_rate:
                continue

            new_label = (new_delay, new_bottle, new_loss_cost)

            # ドミネーションチェック
            is_dominated = False
            for d, b, l in labels[v]:
                if d <= new_delay and b >= new_bottle and l <= new_loss_cost:
                    is_dominated = True
                    break
            if is_dominated:
                continue

            # 既存のドミネートされるラベルを削除
            labels[v] = [(d, b, l) for d, b, l in labels[v] if not (d >= new_delay and b <= new_bottle and l >= new_loss_cost)]
            
            labels[v].append(new_label)
            pred[v][new_label] = (u, current_label)
            heappush(pq, (-new_bottle, new_delay, new_loss_cost, v, new_label))

    final_labels_at_target = labels.get(target, [])
    num_pareto_paths = len(final_labels_at_target)
    
    if not final_labels_at_target:
        return None, -1, -1, -1, num_pareto_paths

    best_bottle = -1
    best_solution_label = None
    for d, b, l in final_labels_at_target:
        # この時点では制約を満たすものしか残っていないはずだが、念のためチェック
        loss_rate = 1 - math.exp(-l)
        if d <= max_delay and loss_rate <= max_loss_rate:
            if b > best_bottle:
                best_bottle = b
                best_solution_label = (d, b, l)

    if best_solution_label is None:
        return None, -1, -1, -1, num_pareto_paths

    # 経路復元
    path = []
    curr_node = target
    curr_label = best_solution_label
    while curr_node is not None:
        path.append(curr_node)
        parent_node, parent_label = pred[curr_node][curr_label]
        curr_node = parent_node
        curr_label = parent_label
    path.reverse()
    
    final_delay, final_bottle, final_loss_cost = best_solution_label
    final_loss_rate = 1 - math.exp(-final_loss_cost)
    
    return path, final_bottle, final_delay, final_loss_rate, num_pareto_paths


def simulation(Graph, src_node, dst_node, delay_multiplier=1.5, loss_rate_constraint=0.1):
    print('始点->終点: ' + str(src_node) + '->' + str(dst_node))

    try:
        min_delay = nx.dijkstra_path_length(Graph, source=src_node, target=dst_node, weight='delay')
        print(f"最短遅延 (Delay-based Dijkstra): {min_delay:.2f}")
    except nx.NetworkXNoPath:
        print("エラー: 始点・終点間に経路が存在しません。このシミュレーションはスキップします。")
        return -1, -1, -1, -1, -1 # ★★★ 戻り値の数を変更 ★★★

    max_allowable_delay = min_delay * delay_multiplier
    print(f"最大許容遅延 (最短遅延の {delay_multiplier} 倍): {max_allowable_delay:.2f}")
    print(f"最大許容ロス率: {loss_rate_constraint:.4f}")

    # --- 比較用ベンチマークはコメントアウト ---
    # # Inverse Dijkstra
    # # Unconstrained Modified Dijkstra
    
    # --- 3基準の厳密解法を実行 ---
    label_correcting_bn = 0
    num_pareto_paths = 0
    lc_start_time = time.time()
    try:
        # ★★★ 3基準対応の関数を呼び出し ★★★
        optimal_path, optimal_bn, optimal_delay, optimal_loss, num_pareto_paths = find_optimal_path_by_label_correcting_3d(
            G=Graph, 
            source=src_node, 
            target=dst_node, 
            max_delay=max_allowable_delay,
            max_loss_rate=loss_rate_constraint
        )
        if optimal_path is not None:
            label_correcting_bn = optimal_bn
            print('<3-criteria Label-Correcting Algorithm (The True Optimal)>')
            print(f'Found {num_pareto_paths} Pareto-optimal paths.')
            print(f'BottleNeck: {optimal_bn:.2f}, Delay: {optimal_delay:.2f}, Loss: {optimal_loss:.4f}, Path: {optimal_path}')
        else:
            print('<3-criteria Label-Correcting Algorithm (The True Optimal)>')
            print(f'No path found within the constraints.')
    except Exception as e:
        print('<3-criteria Label-Correcting Algorithm (The True Optimal)>')
        print(f"An error occurred: {e}")
    lc_end_time = time.time()
    lc_time = lc_end_time - lc_start_time
    print(f"Label-Correcting Execution Time: {lc_time:.4f} sec")
    
    # --- PSOの実行はスキップ ---
    
    # ★★★ 戻り値を厳密解法の結果のみに変更 ★★★
    return label_correcting_bn, num_pareto_paths, lc_time, optimal_delay if 'optimal_delay' in locals() else -1, optimal_loss if 'optimal_loss' in locals() else -1

###############
### main関数 ###
###############
if __name__ == '__main__':
    # ==================================================================
    # ★★★ 実験設定 ★★★
    # ==================================================================
    num_simulation = 5
    node_counts = [3000, 4000] # テストするノード数
    delay_multipliers = [1.5, 3.0]
    loss_rate_constraints = [0.05, 0.1] # ★★★ 調査軸としてロス率制約を追加 (5%と10%) ★★★
    graph_types = ['random', 'ba']

    results = []

    # --- 実験ループ ---
    for num_nodes in node_counts:
        for graph_type in graph_types:
            for multiplier in delay_multipliers:
                for loss_constraint in loss_rate_constraints: # ★★★ ロス率制約のループ ★★★
                    for iter_num in range(num_simulation):
                        print(f'\n--- 試行: {iter_num+1}/{num_simulation} (Node:{num_nodes}, Graph:{graph_type}, Mul:{multiplier}, Loss:{loss_constraint}) ---')
                        
                        if graph_type == 'random':
                            Graph = create_graph_3.rnd_graph(num_node=num_nodes)
                        elif graph_type == 'ba':
                            Graph = create_graph_3.ba_graph(num_node=num_nodes, m=2)

                        if not nx.is_connected(Graph):
                            largest_cc = max(nx.connected_components(Graph), key=len)
                            if len(largest_cc) < 2:
                                print("スキップ: グラフに2つ以上の連結ノードがありません。")
                                continue
                            node_list = list(largest_cc)
                        else:
                            node_list = list(Graph.nodes())
                        
                        src_node = random.choice(node_list)
                        node_list.remove(src_node)
                        dst_node = random.choice(node_list)

                        # ★★★ simulation関数の呼び出しと戻り値を変更 ★★★
                        optimal_bn, num_pareto, lc_time, optimal_delay, optimal_loss = simulation(
                            Graph=Graph,
                            src_node=src_node,
                            dst_node=dst_node,
                            delay_multiplier=multiplier,
                            loss_rate_constraint=loss_constraint
                        )
                        
                        # ★★★ 結果の保存内容を変更 ★★★
                        results.append([
                            num_nodes, iter_num + 1, graph_type, multiplier, loss_constraint,
                            optimal_bn, num_pareto, optimal_delay, optimal_loss, lc_time
                        ])

    # --- CSVファイルへの書き出し ---
    file_name = datetime.now().strftime('%Y%m%d_%H%M%S') + '_3criteria_exact_experiment.csv'
    result_path = savef.create_dir(dir_name="exact_results_3d")
    file_path = result_path + '/' + file_name

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # ★★★ CSVヘッダーを変更 ★★★
        writer.writerow([
            'Num_Nodes', 'iter', 'graph_type', 'delay_multiplier', 'loss_constraint',
            'Optimal_Bottleneck', 'Num_Pareto_Paths', 'Optimal_Delay', 'Optimal_Loss_Rate',
            'Optimal_Time_sec'
        ])
        writer.writerows(results)

    print(f"\n3基準厳密解法の実験結果が '{file_path}' に保存されました。")