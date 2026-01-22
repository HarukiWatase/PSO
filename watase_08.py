import numpy as np
import random
import networkx as nx
import create_graph
from datetime import datetime
import csv
import savef 
from heapq import heappop, heappush
import time

"""
PSO: with comparison (label-correcting algorithm for optimal benchmark)
This script is modified to compare two PSO versions: 'penalty_method' vs 'death_penalty'.
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

def find_optimal_path_by_label_correcting(G, source, target, max_delay, weight="weight", delay="delay"):
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


###########
### 関数 ###
###########

# (PathEncodeからfind_optimal_path_by_label_correctingまでの関数群は変更ありません)
# ...

# ★★★ 変更点：simulation関数はsrc_nodeとdst_nodeを引数で受け取る ★★★
def simulation(Graph, gen, par, src_node, dst_node, delay_multiplier=1.5, use_penalty_method=True):
    num_node = int(len(Graph.nodes()))
    num_par = par
    num_gen = gen
    lb, ub = 1, 20
    # ★★★ 変更点：PSOパラメータを標準的な値に更新 ★★★
    c_1 = 2.0
    c_2 = 2.0
    w_start = 0.9 # 慣性重みの初期値
    w_end = 0.4   # 慣性重みの最終値
    c1_start = 2.5      # c1（個人）の初期値
    c1_end = 0.5        # c1（個人）の最終値
    c2_start = 0.5      # c2（社会）の初期値
    c2_end = 2.5        # c2（社会）の最終値
    # 始点・終点のランダム選択ロジックはmainに移動したため、ここでは削除
    print('始点->終点: ' + str(src_node) + '->' + str(dst_node))

    try:
        min_delay = nx.dijkstra_path_length(Graph, source=src_node, target=dst_node, weight='delay')
        print(f"最短遅延 (Delay-based Dijkstra): {min_delay:.2f}")
    except nx.NetworkXNoPath:
        print("エラー: 始点・終点間に経路が存在しません。このシミュレーションはスキップします。")
        return 0, 0, 0, 0, 0 # 戻り値の数を5つに統一

    max_allowable_delay = min_delay * delay_multiplier
    print(f"最大許容遅延 (最短遅延の {delay_multiplier} 倍): {max_allowable_delay:.2f}")

    label_correcting_bn = 0
    lc_start_time = time.time()
    try:
        _, optimal_bn, _ = find_optimal_path_by_label_correcting(
            Graph, source=src_node, target=dst_node, max_delay=max_allowable_delay
        )
        if optimal_bn != -1:
            label_correcting_bn = optimal_bn
    except Exception:
        pass
    lc_end_time = time.time()
    lc_time = lc_end_time - lc_start_time

    pso_start_time = time.time()
    
    # (PSOのロジックは変更なし)
    # ...
    # ... (gBest_feasibleを含む完全なPSOロジック) ...
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
    
    gBest_feasible_bn = -1.0
    gBest_feasible_path = []
    gBest_feasible_delay = -1.0

    swarms[0] = np.array([[random.uniform(lb, ub) for _ in range(num_node)] for _ in range(num_par)])
    
    path_list, valid_list, bns_list, delay_list = [], [], [], []
    for j in range(num_par):
        temp_path, temp_valid = PathEncode(swarms[0][j], Graph, src=src_node, dst=dst_node)
        path_list.append(temp_path)
        valid_list.append(temp_valid)
        bns_list.append(bottleneck(Graph, temp_path))
        delay_list.append(path2delay(Graph, temp_path))
    paths[0], valids[0], bns[0], delays[0] = path_list, valid_list, bns_list, delay_list

    # ★★★ 変更点：適応度関数のロジックを切り替え可能に ★★★
    for j in range(num_par):
        fitness[0][j] = bns[0][j]
        if not valids[0][j]:
            fitness[0][j] = -1.0
        elif delays[0][j] > max_allowable_delay:
            if use_penalty_method:
                excess_delay = delays[0][j] - max_allowable_delay
                penalty_coefficient = 0.5
                penalty = excess_delay * penalty_coefficient
                fitness[0][j] -= penalty
            else: # 従来の死のペナルティ
                fitness[0][j] = 0

    pBests_fitness[0] = fitness[0]
    pBests[0] = swarms[0]
    temp_index = np.argmax(fitness[0])
    gBests_fitness[0] = fitness[0][temp_index]
    gBests[0] = swarms[0][temp_index]

    for j in range(num_par):
        if valids[0][j] and delays[0][j] <= max_allowable_delay:
            if bns[0][j] > gBest_feasible_bn:
                gBest_feasible_bn = bns[0][j]
                gBest_feasible_path = paths[0][j]
                gBest_feasible_delay = delays[0][j]

    for i in range(1, num_gen):
        # ★★★ 変更点：慣性重みwを線形に減少させる ★★★
        w = w_start - (w_start - w_end) * (i / num_gen)
        c_1 = c1_start - (c1_start - c1_end) * (i / num_gen)
        c_2 = c2_start + (c2_end - c2_start) * (i / num_gen)


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
                if use_penalty_method:
                    excess_delay = delays[i][j] - max_allowable_delay
                    penalty_coefficient = 0.5
                    penalty = excess_delay * penalty_coefficient
                    fitness[i][j] -= penalty
                else: # 従来の死のペナルティ
                    fitness[i][j] = 0

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
    
    return gBest_feasible_bn, gBest_feasible_delay, label_correcting_bn, pso_time, lc_time

###############
### main関数 ###
###############
if __name__ == '__main__':
    num_simulation = 30
    num_nodes = 1000
    num_par = 100
    num_gen = 200

    result_path = savef.create_dir()
    savef.make_doc(result_path, num_node=num_nodes, num_par=num_par, num_gen=num_gen)
    
    delay_multipliers = [1.5, 2.0, 3.0]
    graph_types = ['random']
    pso_versions = ['death_penalty', 'penalty_method']
    
    results = []

    for graph_type in graph_types:
        print(f"\n################### グラフ: {graph_type.upper()} ###################")
        for multiplier in delay_multipliers:
            print(f"\n=============== 遅延倍率: x{multiplier} ===============")
            for iter_num in range(num_simulation):
                print(f'\n--- 試行: {iter_num+1}/{num_simulation} (グラフ: {graph_type}, 倍率: x{multiplier}) ---')
                
                # 最初にグラフを一度だけ生成
                if graph_type == 'random':
                    Graph = create_graph.rnd_graph(num_node=num_nodes)
                elif graph_type == 'grid':
                    Graph = create_graph.grid_graph(num_node=num_nodes)
                elif graph_type == 'ba':
                    Graph = create_graph.ba_graph(num_node=num_nodes, m=2)

                # ★★★ 変更点：ここで始点と終点を一度だけ決定 ★★★
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
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

                for version in pso_versions:
                    print(f'\n>>> PSOバージョン: {version}')
                    
                    is_penalty_method = (version == 'penalty_method')
                    
                    # ★★★ 変更点：決定した始点と終点を引数で渡す ★★★
                    pso_bn, pso_delay, optimal_bn, pso_time, lc_time = simulation(
                        Graph=Graph, gen=num_gen, par=num_par, 
                        src_node=src_node, dst_node=dst_node,
                        delay_multiplier=multiplier, 
                        use_penalty_method=is_penalty_method
                    )
                    
                    results.append([
                        iter_num + 1, 
                        graph_type, 
                        multiplier, 
                        version,
                        pso_bn, 
                        optimal_bn,
                        pso_delay,
                        pso_time, 
                        lc_time
                    ])

    file_name = datetime.now().strftime('%Y%m%d_%H%M%S') + '_pso_comparison.csv'
    file_path = result_path + '/' + file_name

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'iter', 'graph_type', 'delay_multiplier', 'pso_version',
            'PSO_Bottleneck', 'Optimal_Bottleneck',
            'PSO_Delay', 'PSO_Time_sec', 'Optimal_Time_sec'
        ])
        writer.writerows(results)

    print(f"\n比較実験の結果が '{file_path}' に保存されました。")