import numpy as np
import random
import networkx as nx
import create_graph # 3種類のグラフ生成関数を持つcreate_graph.pyを想定
from datetime import datetime
import csv
import savef 
from heapq import heappop, heappush
import time

"""
PSO: with comparison (label-correcting algorithm for optimal benchmark)
This script uses an improved PSO with a PENALTY METHOD for constraint handling.
The output format is kept consistent with the previous version.
"""

###########
### 関数 ###
###########

# (PathEncodeからfind_optimal_path_by_label_correctingまでの関数群は変更ありません)
def PathEncode(Particle, Graph, src, dst):
    path = [src]
    current_node = src
    visited = {src}
    while current_node != dst:
        neighbors = [n for n in Graph.neighbors(current_node) if n not in visited]
        if not neighbors:
            break
        highest_prio = -1
        next_node = -1
        for neighbor in neighbors:
            if Particle[neighbor] > highest_prio:
                highest_prio = Particle[neighbor]
                next_node = neighbor
        if next_node == -1:
            break
        current_node = next_node
        path.append(current_node)
        visited.add(current_node)
    valid = (path[-1] == dst)
    return path, valid

def bottleneck(Graph, Vk):
    if len(Vk) < 2: return 0
    bn = float('inf')
    for i in range(len(Vk)-1):
        if Graph.has_edge(Vk[i], Vk[i+1]) and 'weight' in Graph[Vk[i]][Vk[i+1]]:
            temp = Graph[Vk[i]][Vk[i+1]]['weight']
            if temp < bn:
                bn = temp
        else: return 0
    if bn == float('inf'): return 0
    return bn

def path2delay(Graph, Vk):
    total_delay = 0
    if len(Vk) < 2: return 0
    for i in range(len(Vk) - 1):
        if Graph.has_edge(Vk[i], Vk[i+1]) and 'delay' in Graph[Vk[i]][Vk[i+1]]:
            total_delay += Graph[Vk[i]][Vk[i+1]]['delay']
        else: return float('inf')
    return total_delay

def max_load_path(G, source, target, weight="weight"):
    # (この関数の内容は変更なし)
    if source not in G: raise nx.NodeNotFound(f"Source {source} not in graph")
    if target not in G: raise nx.NodeNotFound(f"Target {target} not in graph")
    pq = []
    bottleneck_vals = {node: float("-inf") for node in G}
    bottleneck_vals[source] = float("inf")
    pred = {node: None for node in G}
    visited = set()
    heappush(pq, (-bottleneck_vals[source], source))
    while pq:
        curr_bottle_neg, u = heappop(pq)
        curr_bottle = -curr_bottle_neg
        if u in visited: continue
        visited.add(u)
        if u == target: break
        for v in G.neighbors(u):
            edge_data = G[u][v]
            w = edge_data.get(weight, 1)
            new_bottle = min(curr_bottle, w)
            if new_bottle > bottleneck_vals[v]:
                bottleneck_vals[v] = new_bottle
                pred[v] = u
                heappush(pq, (-new_bottle, v))
    if bottleneck_vals[target] == float("-inf"): raise nx.NetworkXNoPath(f"No path found from {source} to {target}")
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = pred[current]
    path.reverse()
    return path

def bottleneck_capacity(graph: nx.Graph, path: list) -> float:
    # (この関数の内容は変更なし)
    if not path or len(path) < 2: return 0
    min_capacity = float('inf')
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if graph.has_edge(u, v) and 'weight' in graph[u][v]:
            min_capacity = min(min_capacity, graph[u][v]['weight'])
        else: return 0
    return min_capacity

def find_optimal_path_by_label_correcting(G, source, target, max_delay, weight="weight", delay="delay"):
    # (この関数の内容は変更なし)
    labels = {node: [] for node in G.nodes()}
    pred = {node: {} for node in G.nodes()}
    pq = []
    initial_delay = 0.0
    initial_bottle = float('inf')
    start_label = (initial_delay, initial_bottle)
    labels[source].append(start_label)
    pred[source][start_label] = (None, None)
    heappush(pq, (-initial_bottle, initial_delay, source, start_label))
    while pq:
        neg_bottle, current_delay, u, current_label = heappop(pq)
        current_bottle = -neg_bottle
        if current_label not in labels[u]: continue
        for v in G.neighbors(u):
            edge_data = G.get_edge_data(u, v)
            edge_weight = edge_data.get(weight, 1)
            edge_delay = edge_data.get(delay, 0)
            new_delay = current_delay + edge_delay
            new_bottle = min(current_bottle, edge_weight)
            new_label = (new_delay, new_bottle)
            if new_delay > max_delay: continue
            is_dominated = False
            for d, b in labels[v]:
                if d <= new_delay and b >= new_bottle:
                    is_dominated = True
                    break
            if is_dominated: continue
            labels[v] = [(d, b) for d, b in labels[v] if not (d >= new_delay and b <= new_bottle)]
            labels[v].append(new_label)
            pred[v][new_label] = (u, current_label)
            heappush(pq, (-new_bottle, new_delay, v, new_label))
    final_labels_at_target = labels.get(target, [])
    num_pareto_paths = len(final_labels_at_target)
    if not final_labels_at_target: return None, -1, -1
    best_bottle = -1
    best_solution_label = None
    for d, b in final_labels_at_target:
        if d <= max_delay:
            if b > best_bottle:
                best_bottle = b
                best_solution_label = (d, b)
    if best_solution_label is None: return None, -1, -1
    path = []
    curr_node = target
    curr_label = best_solution_label
    while curr_node is not None:
        path.append(curr_node)
        parent_node, parent_label = pred[curr_node][curr_label]
        curr_node = parent_node
        curr_label = parent_label
    path.reverse()
    final_delay, final_bottle = best_solution_label
    return path, final_bottle, final_delay, num_pareto_paths

# watase_11.py の simulation関数を、以下で置き換えてください

def simulation(Graph, gen, par, src_node, dst_node, delay_multiplier=1.5):
    num_node = int(len(Graph.nodes()))
    num_par = par
    num_gen = gen
    lb, ub = 1, 20

    # PSOパラメータ（慣性重み＋適応的加速係数）
    w_start = 0.9
    w_end = 0.4
    c1_start = 2.5
    c1_end = 0.5
    c2_start = 0.5
    c2_end = 2.5
    # ★★★ 変更点：ペナルティ係数を動的にするための設定を追加 ★★★
    penalty_coeff_start = 0.1 # ペナルティの初期の重み（軽く）
    penalty_coeff_end = 1.0   # ペナルティの最終的な重み（重く）



    print('始点->終点: ' + str(src_node) + '->' + str(dst_node))

    try:
        min_delay = nx.dijkstra_path_length(Graph, source=src_node, target=dst_node, weight='delay')
        print(f"最短遅延 (Delay-based Dijkstra): {min_delay:.2f}")
    except nx.NetworkXNoPath:
        print("エラー: 始点・終点間に経路が存在しません。このシミュレーションはスキップします。")
        return 0, 0, 0, 0, 0, 0, 0

    max_allowable_delay = min_delay * delay_multiplier
    print(f"最大許容遅延 (最短遅延の {delay_multiplier} 倍): {max_allowable_delay:.2f}")

    # --- 比較用ベンチマークの実行 ---
    # Inverse Dijkstra
    for (i,j) in Graph.edges():
        if 'weight' in Graph[i][j]:
            temp_inv = 1/Graph[i][j]["weight"] if Graph[i][j]["weight"] != 0 else float('inf')
            Graph[i][j]['inv'] = temp_inv
        else:
            Graph[i][j]['inv'] = float('inf')
    bn_inv = 0
    try:
        v_inv = nx.dijkstra_path(Graph, source=src_node, target=dst_node, weight='inv')
        bn_inv = bottleneck(Graph, v_inv)
        print('<Inverse Dijkstra>')
        print(f'BottleNeck: {bn_inv}, path: {v_inv}')
    except nx.NetworkXNoPath:
        print('<Inverse Dijkstra>')
        print('No path found.')

    # Unconstrained Modified Dijkstra
    mod_dijkstra_bn = 0
    try:
        v_mod_dijkstra = max_load_path(Graph, source=src_node, target=dst_node, weight='weight')
        mod_dijkstra_bn = bottleneck_capacity(Graph, v_mod_dijkstra)
        print('<Unconstrained Modified Dijkstra>')
        print(f'BottleNeck: {mod_dijkstra_bn}, path: {v_mod_dijkstra}')
    except nx.NetworkXNoPath:
        print('<Unconstrained Modified Dijkstra>')
        print('No path found.')

    
    # Label-Correcting Algorithm
    label_correcting_bn = 0
    num_pareto_paths = 0
    lc_start_time = time.time()
    try:
        optimal_path, optimal_bn, optimal_delay, num_pareto_paths = find_optimal_path_by_label_correcting(
            Graph, source=src_node, target=dst_node, max_delay=max_allowable_delay
        )
        if optimal_path is not None:
            label_correcting_bn = optimal_bn
            print('<Label-Correcting Algorithm (The True Optimal)>')
            print(f'Found {num_pareto_paths} Pareto-optimal paths.')
            print(f'BottleNeck: {optimal_bn:.2f}, Delay: {optimal_delay:.2f}, Path: {optimal_path}')
        else:
            print('<Label-Correcting Algorithm (The True Optimal)>')
            print(f'No path found within delay {max_allowable_delay:.2f}')
    except Exception as e:
        print('<Label-Correcting Algorithm (The True Optimal)>')
        print(f"An error occurred: {e}")
    lc_end_time = time.time()
    lc_time = lc_end_time - lc_start_time
    print(f"Label-Correcting Execution Time: {lc_time:.4f} sec")
    
    # --- PSOの実行 ---
    pso_start_time = time.time()
    
    # ★★★ 変更点：PSO変数を2次元配列（現在世代分のみ）に変更 ★★★
    swarms = np.array([[random.uniform(lb, ub) for _ in range(num_node)] for _ in range(num_par)])
    velocities = np.zeros_like(swarms)
    pBests = np.copy(swarms)
    
    # --- 0世代目の評価 ---
    paths_gen0, valids_gen0, bns_gen0, delays_gen0 = [], [], [], []
    for j in range(num_par):
        path, valid = PathEncode(swarms[j], Graph, src=src_node, dst=dst_node)
        paths_gen0.append(path)
        valids_gen0.append(valid)
        bns_gen0.append(bottleneck(Graph, path))
        delays_gen0.append(path2delay(Graph, path))

    fitness = np.zeros(num_par)
    for j in range(num_par):
        fitness[j] = bns_gen0[j]
        if not valids_gen0[j]:
            fitness[j] = -1.0
        elif delays_gen0[j] > max_allowable_delay:
            excess_delay = delays_gen0[j] - max_allowable_delay
            penalty = excess_delay * 0.5
            fitness[j] -= penalty
    
    pBests_fitness = np.copy(fitness)
    
    gBest_index = np.argmax(fitness)
    gBest = np.copy(swarms[gBest_index])
    gBest_fitness = fitness[gBest_index]

    gBest_feasible_bn = -1.0
    gBest_feasible_path = []
    gBest_feasible_delay = -1.0
    for j in range(num_par):
        if valids_gen0[j] and delays_gen0[j] <= max_allowable_delay:
            if bns_gen0[j] > gBest_feasible_bn:
                gBest_feasible_bn = bns_gen0[j]
                gBest_feasible_path = paths_gen0[j]
                gBest_feasible_delay = delays_gen0[j]
                
    # 最適化ループ
    for i in range(1, num_gen):
        w = w_start - (w_start - w_end) * (i / num_gen)
        c1 = c1_start - (c1_start - c1_end) * (i / num_gen)
        c2 = c2_start + (c2_end - c2_start) * (i / num_gen)
         # ★★★ 変更点：ペナルティ係数を世代ごとに更新 ★★★
        penalty_coefficient = penalty_coeff_start + (penalty_coeff_end - penalty_coeff_start) * (i / num_gen)

        r_1 = np.random.rand(num_par, 1)
        r_2 = np.random.rand(num_par, 1)
        
        velocities = w * velocities + c1 * r_1 * (pBests - swarms) + c2 * r_2 * (gBest - swarms)
        swarms = swarms + velocities
        
        paths_current_gen, valids_current_gen, bns_current_gen, delays_current_gen = [], [], [], []
        for j in range(num_par):
            path, valid = PathEncode(swarms[j], Graph, src=src_node, dst=dst_node)
            paths_current_gen.append(path)
            valids_current_gen.append(valid)
            bns_current_gen.append(bottleneck(Graph, path))
            delays_current_gen.append(path2delay(Graph, path))

        for j in range(num_par):
            fitness[j] = bns_current_gen[j]
            if not valids_current_gen[j]:
                fitness[j] = -1.0
            elif delays_current_gen[j] > max_allowable_delay:
                excess_delay = delays_current_gen[j] - max_allowable_delay
                penalty = excess_delay * penalty_coefficient
                fitness[j] -= penalty

        update_indices = fitness > pBests_fitness
        pBests[update_indices] = swarms[update_indices]
        pBests_fitness[update_indices] = fitness[update_indices]
        
        current_best_idx = np.argmax(fitness)
        if fitness[current_best_idx] > gBest_fitness:
            gBest_fitness = fitness[current_best_idx]
            gBest = swarms[current_best_idx]

        for j in range(num_par):
            if valids_current_gen[j] and delays_current_gen[j] <= max_allowable_delay:
                if bns_current_gen[j] > gBest_feasible_bn:
                    gBest_feasible_bn = bns_current_gen[j]
                    gBest_feasible_path = paths_current_gen[j]
                    gBest_feasible_delay = delays_current_gen[j]

    pso_end_time = time.time()
    pso_time = pso_end_time - pso_start_time
    
    print('<PSO Feasible Solution>')
    print(f'BottleNeck: {gBest_feasible_bn:.2f}, path: {gBest_feasible_path}, delay: {gBest_feasible_delay:.2f}')
    print(f"PSO Execution Time: {pso_time:.4f} sec")
    
    # ★★★ 戻り値の形式はmain関数に合わせて維持 ★★★
    return gBest_feasible_bn, bn_inv, gBest_feasible_delay, mod_dijkstra_bn, label_correcting_bn, num_pareto_paths, pso_time, lc_time
###############
### main関数 ###
###############
if __name__ == '__main__':
    # ★★★ 変更点：テストしたいノード数のリストを定義 ★★★
    node_counts = [100,300,500,700,900,1100,1300,1500,1700,1900,2100,2300,2500,2700,2900] 

    # 固定パラメータ
    num_simulation = 50
    num_par = 300
    num_gen = 500
    delay_multipliers = [1.5,2.0,2.5]
    graph_types = ['ba']
    pso_versions = ['penalty_method']
    
    # 全ての結果を格納するリスト
    results = []

    # ★★★ 変更点：ノード数ごとにループを追加 ★★★
    for num_nodes in node_counts:
        print(f"\n============================================================")
        print(f"========== ノード数: {num_nodes} でのシミュレーションを開始 ==========")
        print(f"============================================================")

        for graph_type in graph_types:
            print(f"\n################### グラフ: {graph_type.upper()} ###################")
            for multiplier in delay_multipliers:
                print(f"\n=============== 遅延倍率: x{multiplier} ===============")
                for iter_num in range(num_simulation):
                    print(f'\n--- 試行: {iter_num+1}/{num_simulation} (ノード数: {num_nodes}, グラフ: {graph_type}, 倍率: x{multiplier}) ---')
                    
                    if graph_type == 'random':
                        Graph = create_graph.rnd_graph(num_node=num_nodes)
                    elif graph_type == 'grid':
                        Graph = create_graph.grid_graph(num_node=num_nodes)
                    elif graph_type == 'ba':
                        Graph = create_graph.ba_graph(num_node=num_nodes, m=2)

                    src_node, dst_node = None, None # 初期化
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

                    for version in pso_versions:
                        print(f'\n>>> PSOバージョン: {version}')
                        
                        is_penalty_method = (version == 'penalty_method')
                        
                        pso_bn, inv_dijkstra_bn, pso_delay, mod_dijkstra_bn, label_correcting_bn, num_pareto, pso_time, lc_time = simulation(
                            Graph=Graph, gen=num_gen, par=num_par, 
                            src_node=src_node, dst_node=dst_node,
                            delay_multiplier=multiplier, 
                        )
                        
                        # ★★★ 変更点：結果リストにノード数を追加 ★★★
                        results.append([
                            num_nodes, # ノード数を記録
                            iter_num + 1, 
                            graph_type, 
                            multiplier, 
                            pso_bn, 
                            inv_dijkstra_bn,
                            mod_dijkstra_bn,
                            label_correcting_bn, 
                            num_pareto,
                            pso_delay,
                            pso_time, 
                            lc_time,
                        ])

    # --- 全ての実験が完了した後に、一度だけCSVファイルに書き出す ---
    file_name = datetime.now().strftime('%Y%m%d_%H%M%S') + '_scalability_experiment.csv'
    # 総合的な結果を保存するメインのパス
    final_result_path = savef.create_dir(dir_name="final_results")
    file_path = final_result_path + '/' + file_name

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # ★★★ 変更点：CSVヘッダーにノード数の列を追加 ★★★
        writer.writerow([
            'Num_Nodes', 'iter', 'graph_type', 'delay_multiplier', 'PSO_Bottleneck',
            'Inverse_Dijkstra_BN', 'Unconstrained_Mod_Dijkstra_BN', 'Optimal_Bottleneck',
            'Num_Pareto_Paths', 'PSO_Delay', 
            'PSO_Time_sec', 'Optimal_Time_sec', 
        ])
        writer.writerows(results)

    print(f"\n全てのシミュレーションが完了しました。")
    print(f"結果が '{file_path}' に保存されました。")