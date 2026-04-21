# watase_17_label.py

import networkx as nx
import create_graph_4 as create_graph
from datetime import datetime
import csv
import savef
from heapq import heappop, heappush
import time
import math
import random
"""
This script runs a 4-criteria label-correcting algorithm.
(Bottleneck, Delay, Loss Rate, Reliability)
The output now includes the optimal path.
"""

###########
### 関数 ###
###########

def find_optimal_path_by_label_correcting_4d(G, source, target, max_delay, max_loss_rate, min_reliability):
    # (この関数に変更はありません)
    labels = {node: [] for node in G.nodes()}
    pred = {node: {} for node in G.nodes()}
    pq = []
    initial_label = (0.0, float('inf'), 0.0, 0.0)
    labels[source].append(initial_label)
    pred[source][initial_label] = (None, None)
    heappush(pq, (-initial_label[1], initial_label[0], initial_label[2], initial_label[3], source, initial_label))
    min_reliability_cost = -math.log(min_reliability)

    while pq:
        neg_bottle, d_curr, l_curr, r_curr, u, label_curr = heappop(pq)
        if label_curr not in labels[u]:
            continue
        for v in G.neighbors(u):
            edge = G.get_edge_data(u, v)
            d_new = d_curr + edge.get("delay", 0)
            b_new = min(-neg_bottle, edge.get("weight", 1))
            l_new = l_curr + edge.get("loss_log_cost", 0)
            r_new = r_curr + edge.get("reliability_cost", 0)
            if d_new > max_delay: continue
            if l_new > -math.log(1 - max_loss_rate): continue
            if r_new > min_reliability_cost: continue
            label_new = (d_new, b_new, l_new, r_new)
            is_dominated = False
            for d, b, l, r in labels[v]:
                if d <= d_new and b >= b_new and l <= l_new and r <= r_new:
                    is_dominated = True
                    break
            if is_dominated:
                continue
            labels[v] = [(d, b, l, r) for d, b, l, r in labels[v] if not (d >= d_new and b <= b_new and l >= l_new and r >= r_new)]
            labels[v].append(label_new)
            pred[v][label_new] = (u, label_curr)
            heappush(pq, (-b_new, d_new, l_new, r_new, v, label_new))

    final_labels = labels.get(target, [])
    num_pareto_paths = len(final_labels)
    if not final_labels:
        return None, -1, -1, -1, -1, num_pareto_paths

    best_bottle = -1
    best_solution = None
    for d, b, l, r in final_labels:
        if b > best_bottle:
            best_bottle = b
            best_solution = (d, b, l, r)

    if best_solution is None:
        return None, -1, -1, -1, -1, num_pareto_paths

    path = []
    curr_node, curr_label = target, best_solution
    while curr_node is not None:
        path.append(curr_node)
        curr_node, curr_label = pred[curr_node][curr_label]
    path.reverse()
    
    final_d, final_b, final_l, final_r = best_solution
    final_loss = 1 - math.exp(-final_l)
    final_reliability = math.exp(-final_r)
    
    return path, final_b, final_d, final_loss, final_reliability, num_pareto_paths


def simulation(Graph, src_node, dst_node, delay_multiplier=1.5, loss_constraint=0.1, reliability_constraint=0.99):
    print(f'始点->終点: {src_node}->{dst_node}')

    try:
        min_delay = nx.dijkstra_path_length(Graph, source=src_node, target=dst_node, weight='delay')
    except nx.NetworkXNoPath:
        print("エラー: 経路が存在しません。スキップします。")
        return None, -1, -1, -1, -1, -1, -1 # ★★★ 戻り値の数を変更 ★★★

    max_allowable_delay = min_delay * delay_multiplier
    print(f"最大許容遅延: {max_allowable_delay:.2f}, 最大許容ロス率: {loss_constraint:.4f}, 最小要求信頼度: {reliability_constraint:.5f}")
    
    lc_start_time = time.time()
    optimal_path = None # 初期化
    try:
        optimal_path, bn, d, l, r, num_pareto = find_optimal_path_by_label_correcting_4d(
            G=Graph, source=src_node, target=dst_node, 
            max_delay=max_allowable_delay,
            max_loss_rate=loss_constraint,
            min_reliability=reliability_constraint
        )
        if optimal_path:
            print(f'<4-criteria Label-Correcting Algorithm>')
            print(f'Found {num_pareto} Pareto-optimal paths.')
            print(f'BN: {bn:.2f}, Delay: {d:.2f}, Loss: {l:.4f}, Reliability: {r:.5f}')
            # ★★★ ここに経路表示を追加 ★★★
            print(f'Path: {optimal_path}')
        else:
            print(f'<4-criteria Label-Correcting Algorithm>: No path found within constraints.')
    except Exception as e:
        print(f"An error occurred: {e}")
    lc_end_time = time.time()
    lc_time = lc_end_time - lc_start_time
    print(f"Execution Time: {lc_time:.4f} sec")
    
    # ★★★ 戻り値に optimal_path を追加 ★★★
    return optimal_path, bn if 'bn' in locals() else -1, num_pareto if 'num_pareto' in locals() else -1, lc_time, d if 'd' in locals() else -1, l if 'l' in locals() else -1, r if 'r' in locals() else -1

###############
### main関数 ###
###############
if __name__ == '__main__':
    num_simulation = 3
    node_counts = [3000, 4000]
    delay_multipliers = [3.0]
    loss_rate_constraints = [0.5] # 許容ロス率を20%や30%に緩和
    reliability_constraints = [0.70] # 要求信頼度を95%や90%に緩和
    graph_types = ['random', 'ba']
    results = []

    for num_nodes in node_counts:
        for graph_type in graph_types:
            for multiplier in delay_multipliers:
                for loss_con in loss_rate_constraints:
                    for rel_con in reliability_constraints:
                        for i in range(num_simulation):
                            print(f'\n--- Sim: {i+1}/{num_simulation} (N:{num_nodes}, G:{graph_type}, D:{multiplier}, L:{loss_con}, R:{rel_con}) ---')
                            
                            Graph = getattr(create_graph, f"{graph_type}_graph")(num_node=num_nodes)
                            
                            if not nx.is_connected(Graph):
                                largest_cc = max(nx.connected_components(Graph), key=len)
                                if len(largest_cc) < 2: continue
                                node_list = list(largest_cc)
                            else:
                                node_list = list(Graph.nodes())
                            src, dst = random.sample(node_list, 2)

                            # ★★★ 戻り値に path を追加 ★★★
                            path, bn, num_pareto, lc_time, d, l, r = simulation(
                                Graph=Graph, src_node=src, dst_node=dst,
                                delay_multiplier=multiplier,
                                loss_constraint=loss_con,
                                reliability_constraint=rel_con
                            )
                            # ★★★ 結果の保存に path を追加 ★★★
                            # CSVに保存する際、リストを文字列に変換
                            path_str = '->'.join(map(str, path)) if path else 'N/A'
                            results.append([
                                num_nodes, i + 1, graph_type, multiplier, loss_con, rel_con,
                                bn, num_pareto, d, l, r, lc_time, path_str
                            ])

    file_name = datetime.now().strftime('%Y%m%d_%H%M%S') + '_4criteria_exact_experiment.csv'
    result_path = savef.create_dir(dir_name="exact_results_4d")
    file_path = result_path + '/' + file_name

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # ★★★ CSVヘッダーに Path を追加 ★★★
        writer.writerow([
            'Num_Nodes', 'iter', 'graph_type', 'delay_multiplier', 'loss_constraint', 'reliability_constraint',
            'Optimal_BN', 'Num_Pareto', 'Optimal_Delay', 'Optimal_Loss', 'Optimal_Reliability', 'Optimal_Time_sec',
            'Optimal_Path'
        ])
        writer.writerows(results)

    print(f"\n4基準厳密解法の実験結果が '{file_path}' に保存されました。")