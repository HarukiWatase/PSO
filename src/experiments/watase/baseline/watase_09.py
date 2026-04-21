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
    return path, final_bottle, final_delay

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★ simulation関数を「ペナルティ法＋実行可能最良解記録」方式に全面的に更新 ★★★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
def simulation(Graph, gen, par, delay_multiplier=1.5):
    num_node = int(len(Graph.nodes()))
    num_par = par
    num_gen = gen
    lb, ub = 1, 20
    # ★★★ 変更点：PSOパラメータを標準的な値に更新 ★★★
    c_1 = 2.0
    c_2 = 2.0
    w_start = 0.9 # 慣性重みの初期値
    w_end = 0.4   # 慣性重みの最終値

    if not nx.is_connected(Graph):
        largest_cc = max(nx.connected_components(Graph), key=len)
        if len(largest_cc) < 2:
            print("エラー: 2つ以上のノードを持つ連結成分がありません。")
            return 0, 0, 0, 0, 0, 0, 0
        node_list = list(largest_cc)
    else:
        node_list = list(range(num_node))

    src_node = random.choice(node_list)
    node_list.remove(src_node)
    dst_node = random.choice(node_list)
    
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
    lc_start_time = time.time()
    try:
        optimal_path, optimal_bn, optimal_delay = find_optimal_path_by_label_correcting(
            Graph, source=src_node, target=dst_node, max_delay=max_allowable_delay
        )
        if optimal_path is not None:
            label_correcting_bn = optimal_bn
            print('<Label-Correcting Algorithm (The True Optimal)>')
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
    
    swarms = np.zeros((num_gen, num_par, num_node))
    velocities = np.zeros_like(swarms)
    paths = [[] for _ in range(num_gen)]
    valids = [[] for _ in range(num_gen)]
    bns = np.zeros((num_gen, num_par))
    delays = np.zeros((num_gen, num_par))
    fitness = np.zeros_like(bns)
    pBests_fitness = np.zeros_like(bns)
    pBests = np.zeros_like(swarms)
    gBests_fitness = np.zeros(num_gen)
    gBests = np.zeros((num_gen, num_node))
    
    # 実行可能解の中で最良のものを記録する変数
    gBest_feasible_bn = -1.0
    gBest_feasible_path = []
    gBest_feasible_delay = -1.0

    # PSO初期化
    swarms[0] = np.array([[random.uniform(lb, ub) for _ in range(num_node)] for _ in range(num_par)])
    
    path_list, valid_list, bns_list, delay_list = [], [], [], []
    for j in range(num_par):
        temp_path, temp_valid = PathEncode(swarms[0][j], Graph, src=src_node, dst=dst_node)
        path_list.append(temp_path)
        valid_list.append(temp_valid)
        bns_list.append(bottleneck(Graph, temp_path))
        delay_list.append(path2delay(Graph, temp_path))
    paths[0], valids[0], bns[0], delays[0] = path_list, valid_list, bns_list, delay_list

    # 0世代目の適応度計算（ペナルティ法）
    for j in range(num_par):
        fitness[0][j] = bns[0][j]
        if not valids[0][j]:
            fitness[0][j] = -1.0
        elif delays[0][j] > max_allowable_delay:
            excess_delay = delays[0][j] - max_allowable_delay
            penalty_coefficient = 0.5
            penalty = excess_delay * penalty_coefficient
            fitness[0][j] -= penalty

    pBests_fitness[0] = fitness[0]
    pBests[0] = swarms[0]

    temp_index = np.argmax(fitness[0])
    gBests_fitness[0] = fitness[0][temp_index]
    gBests[0] = swarms[0][temp_index]

    # 0世代目のgBest_feasibleを初期化
    for j in range(num_par):
        if valids[0][j] and delays[0][j] <= max_allowable_delay:
            if bns[0][j] > gBest_feasible_bn:
                gBest_feasible_bn = bns[0][j]
                gBest_feasible_path = paths[0][j]
                gBest_feasible_delay = delays[0][j]

    # 最適化ループ
    for i in range(1, num_gen):
        # ★★★ 変更点：慣性重みwを線形に減少させる ★★★
        w = w_start - (w_start - w_end) * (i / num_gen)
        
        r_1, r_2 = random.random(), random.random()
        
        # ★★★ 変更点：速度更新式に慣性重みwを適用 ★★★
        velocities[i] = (w * velocities[i-1] + 
                         c_1 * r_1 * (pBests[i-1] - swarms[i-1]) + 
                         c_2 * r_2 * (gBests[i-1] - swarms[i-1]))
        
        swarms[i] = swarms[i-1] + velocities[i]
        
        path_list, valid_list, bns_list, delay_list = [], [], [], []
        for j in range(num_par):
            temp_path, temp_valid = PathEncode(swarms[i][j], Graph, src=src_node, dst=dst_node)
            path_list.append(temp_path)
            valid_list.append(temp_valid)
            bns_list.append(bottleneck(Graph, temp_path))
            delay_list.append(path2delay(Graph, temp_path))
        paths[i], valids[i], bns[i], delays[i] = path_list, valid_list, bns_list, delay_list

        for j in range(num_par):
            fitness[i][j] = bns[i][j]
            if not valids[i][j]:
                fitness[i][j] = -1.0
            elif delays[i][j] > max_allowable_delay:
                excess_delay = delays[i][j] - max_allowable_delay
                penalty_coefficient = 0.5
                penalty = excess_delay * penalty_coefficient
                fitness[i][j] -= penalty

        for j in range(num_par):
            if pBests_fitness[i-1][j] < fitness[i][j]:
                pBests_fitness[i][j] = fitness[i][j]
                pBests[i][j] = swarms[i][j]
            else:
                pBests_fitness[i][j] = pBests_fitness[i-1][j]
                pBests[i][j] = pBests[i-1][j]
                
        if np.max(fitness[i]) > gBests_fitness[i-1]:
            temp_index = np.argmax(fitness[i])
            gBests_fitness[i] = fitness[i][temp_index]
            gBests[i] = swarms[i][temp_index]
        else:
            gBests_fitness[i] = gBests_fitness[i-1]
            gBests[i] = gBests[i-1]

        current_gen_best_feasible_bn = -1.0
        best_feasible_particle_index = -1
        for j in range(num_par):
            if valids[i][j] and delays[i][j] <= max_allowable_delay:
                if bns[i][j] > current_gen_best_feasible_bn:
                    current_gen_best_feasible_bn = bns[i][j]
                    best_feasible_particle_index = j
        
        if best_feasible_particle_index != -1:
            if current_gen_best_feasible_bn > gBest_feasible_bn:
                gBest_feasible_bn = current_gen_best_feasible_bn
                gBest_feasible_path = paths[i][best_feasible_particle_index]
                gBest_feasible_delay = delays[i][best_feasible_particle_index]

    pso_end_time = time.time()
    pso_time = pso_end_time - pso_start_time
    
    print('<PSO Feasible Solution>')
    print(f'BottleNeck: {gBest_feasible_bn:.2f}, path: {gBest_feasible_path}, delay: {gBest_feasible_delay:.2f}')
    print(f"PSO Execution Time: {pso_time:.4f} sec")
    
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★ 変更点：戻り値の数を7つに戻します ★★★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    return gBest_feasible_bn, bn_inv, gBest_feasible_delay, mod_dijkstra_bn, label_correcting_bn, pso_time, lc_time

###############
### main関数 ###
###############
if __name__ == '__main__':
    # (main関数は変更なし)
    num_simulation = 3
    num_nodes = 100
    num_par = 200
    num_gen = 200

    result_path = savef.create_dir()
    savef.make_doc(result_path, num_node=num_nodes, num_par=num_par, num_gen=num_gen)
    
    delay_multipliers = [1.2, 1.5]
    graph_types = ['random', 'grid', 'ba']
    results = []

    for graph_type in graph_types:
        print(f"\n############################################################")
        print(f"##### グラフタイプ: {graph_type.upper()} でのシミュレーションを開始 #####")
        print(f"############################################################")
        for multiplier in delay_multipliers:
            print(f"\n========================================================")
            print(f"===== 遅延倍率: x{multiplier} でのシミュレーションを開始 =====")
            print(f"========================================================")
            for iter_num in range(num_simulation):
                print(f'\n--- 試行回数: {iter_num+1}/{num_simulation} (グラフ: {graph_type}, 倍率: x{multiplier}) ---')
                
                if graph_type == 'random':
                    Graph = create_graph.rnd_graph(num_node=num_nodes)
                elif graph_type == 'grid':
                    Graph = create_graph.grid_graph(num_node=num_nodes)
                elif graph_type == 'ba':
                    Graph = create_graph.ba_graph(num_node=num_nodes, m=2)

                pso_bn, inv_dijkstra_bn, pso_delay, mod_dijkstra_bn, label_correcting_bn, pso_time, lc_time = simulation(
                    Graph=Graph, gen=num_gen, par=num_par, delay_multiplier=multiplier
                )
                
                results.append([
                    iter_num + 1, graph_type, multiplier, 
                    pso_bn, inv_dijkstra_bn,
                    mod_dijkstra_bn, label_correcting_bn, pso_delay,
                    pso_time, lc_time
                ])

    file_name = datetime.now().strftime('%Y%m%d_%H%M%S') + '_improved_pso.csv'
    file_path = result_path + '/' + file_name

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'iter', 'graph_type', 'delay_multiplier',
            'PSO_Bottleneck', 'Inverse_Dijkstra_BN',
            'Unconstrained_Mod_Dijkstra_BN', 'Optimal_Bottleneck',
            'PSO_Delay', 'PSO_Time_sec', 'Optimal_Time_sec'
        ])
        writer.writerows(results)

    print(f"\n結果が '{file_path}' に保存されました。")