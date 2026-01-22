# watase_28_stochastic.py
# (ベース: watase_26_gen_log.py)

import networkx as nx
import numpy as np
import create_graph_4 as create_graph
from datetime import datetime
import csv
import savef
from heapq import heappop, heappush
import time
import math
import random

"""
[変更点 (watase_28)]
- 「最後の一手」：確率的PathEncodeの導入
- PathEncode が、優先度最大のノードを「必ず」選ぶのではなく、
  優先度に基づいた「確率（ルーレット選択）」で
  次のノードを選ぶように変更。
- これにより、決定論的な局所解の罠からの脱出を狙う。
- (watase_27 の経路修復ロジックは削除し、
   watase_26 の is_valid=False -> Fitness=-1.0 に戻しています)
"""

# (calculate_path_attributes_4d, find_optimal_path_by_label_correcting_4d は watase_26 と同一)
def calculate_path_attributes_4d(G, path):
    # (watase_26_gen_log.py と同一)
    if not path or len(path) < 2: return 0, float('inf'), 1.0, 0.0
    bottleneck = float('inf'); total_delay = 0
    total_loss_log_cost = 0; total_reliability_cost = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if not G.has_edge(u, v): return 0, float('inf'), 1.0, 0.0
        edge_data = G.edges[u, v]
        bottleneck = min(bottleneck, edge_data.get('weight', 0))
        total_delay += edge_data.get('delay', float('inf'))
        total_loss_log_cost += edge_data.get('loss_log_cost', float('inf'))
        total_reliability_cost += edge_data.get('reliability_cost', float('inf'))
    total_loss_rate = 1 - math.exp(-total_loss_log_cost)
    total_reliability = math.exp(-total_reliability_cost)
    return bottleneck, total_delay, total_loss_rate, total_reliability

def find_optimal_path_by_label_correcting_4d(G, source, target, max_delay, max_loss_rate, min_reliability):
    # (watase_26_gen_log.py と同一)
    labels = {node: [] for node in G.nodes()}; pred = {node: {} for node in G.nodes()}; pq = []
    initial_label = (0.0, float('inf'), 0.0, 0.0) # d, b, l, r
    labels[source].append(initial_label); pred[source][initial_label] = (None, None)
    heappush(pq, (-initial_label[1], initial_label[0], initial_label[2], initial_label[3], source, initial_label))
    min_reliability_cost = -math.log(min_reliability) if min_reliability > 0 else float('inf')
    max_loss_log_cost = -math.log(1 - max_loss_rate) if max_loss_rate < 1 else float('inf')

    while pq:
        neg_bottle, d_curr, l_curr, r_curr, u, label_curr = heappop(pq)
        if label_curr not in labels[u]: continue
        for v in G.neighbors(u):
            edge = G.get_edge_data(u, v)
            d_new = d_curr + edge.get("delay", 0); b_new = min(-neg_bottle, edge.get("weight", 1))
            l_new = l_curr + edge.get("loss_log_cost", 0); r_new = r_curr + edge.get("reliability_cost", 0)
            if d_new > max_delay: continue
            if l_new > max_loss_log_cost: continue 
            if r_new > min_reliability_cost: continue
            label_new = (d_new, b_new, l_new, r_new)
            is_dominated = False
            for d, b, l, r in labels[v]:
                if d <= d_new and b >= b_new and l <= l_new and r <= r_new: is_dominated = True; break
            if is_dominated: continue
            labels[v] = [(d, b, l, r) for d, b, l, r in labels[v] if not (d >= d_new and b <= b_new and l >= l_new and r >= r_new)]
            labels[v].append(label_new); pred[v][label_new] = (u, label_curr)
            heappush(pq, (-b_new, d_new, l_new, r_new, v, label_new))

    final_labels = labels.get(target, []); num_pareto_paths = len(final_labels)
    if not final_labels: return None, -1, -1, -1, -1, num_pareto_paths
    best_bottle = -1; best_solution = None
    for d, b, l, r in final_labels:
        if b > best_bottle: best_bottle = b; best_solution = (d, b, l, r)
    if best_solution is None: return None, -1, -1, -1, -1, num_pareto_paths
    path = []; curr_node, curr_label = target, best_solution
    while curr_node is not None: path.append(curr_node); curr_node, curr_label = pred[curr_node][curr_label]
    path.reverse()
    final_d, final_b, final_l, final_r = best_solution
    final_loss = 1 - math.exp(-final_l); final_reliability = math.exp(-final_r)
    return path, final_b, final_d, final_loss, final_reliability, num_pareto_paths


# ▼▼▼ ★★★ 変更点: 確率的 PathEncode ★★★ ▼▼▼
def PathEncode(Particle, Graph, src, dst, temperature=1.0):
    """
    確率的 (Stochastic) Path Encoding
    - 優先度最大のノードを「必ず」選ぶ (Greedy) のをやめる
    - 優先度を Softmax (または正規化) して「確率」に変換し、
      ルーレット選択で次のノードを決定する
    """
    path = [src]
    current_node = src
    visited = {src}
    particle_values = Particle 
    
    while current_node != dst:
        neighbors = [n for n in Graph.neighbors(current_node) if n not in visited]
        if not neighbors: 
            return path, False # 行き止まり

        # --- 確率的選択ロジック ---
        neighbor_priorities = []
        valid_neighbors = []
        
        for neighbor in neighbors:
            if neighbor < len(particle_values): # インデックスチェック
                # 優先度 (Particleの値) は通常 1.0 より大きい
                # 0以下の値があると確率計算で問題が起きるため、最小値を 0.1 等にクリップ
                priority = max(particle_values[neighbor] / temperature, 0.1) 
                neighbor_priorities.append(priority)
                valid_neighbors.append(neighbor)

        if not valid_neighbors:
             return path, False # 有効な隣接ノードがいない

        # 優先度を確率に変換 (Softmaxではなく単純な正規化)
        total_priority = sum(neighbor_priorities)
        probabilities = [p / total_priority for p in neighbor_priorities]
        
        # 確率に基づいて次のノードを1つ選択
        next_node = random.choices(valid_neighbors, weights=probabilities, k=1)[0]
        # --- 変更ここまで ---

        current_node = next_node
        path.append(current_node)
        visited.add(current_node)
        
    return path, path[-1] == dst

# ▼▼▼ ★★★ 変更点: simulation 関数を (watase_26 ベースに) 戻す ★★★ ▼▼▼
def simulation(Graph, src, dst, constraints, pso_params):
    # (厳密解法の実行部分は watase_26_gen_log.py と同一)
    try: 
        min_delay_path_dijkstra = nx.dijkstra_path(Graph, source=src, target=dst, weight='delay')
        _, min_delay, _, _ = calculate_path_attributes_4d(Graph, min_delay_path_dijkstra)
    except nx.NetworkXNoPath: 
        print("Error: No path exists between source and target.")
        return (None,) * 10
        
    max_delay = min_delay * constraints['delay_multiplier']
    max_loss = constraints['loss_constraint']
    min_rel = constraints['reliability_constraint']
    lc_start_time = time.time()
    opt_path, opt_bn, _, _, _, _ = find_optimal_path_by_label_correcting_4d(Graph, src, dst, max_delay, max_loss, min_rel)
    lc_time = time.time() - lc_start_time

    # --- PSOの実行 ---
    pso_start_time = time.time()
    pso_generation_history = [] 

    num_nodes, num_par, num_gen = len(Graph.nodes()), pso_params['num_par'], pso_params['num_gen']
    w_start, w_end = pso_params['w_config']; c1_start, c1_end = pso_params['c1_config']
    c2_start, c2_end = pso_params['c2_config']; Pd_start, Pd_end = pso_params['Pd_config']
    Pl_start, Pl_end = pso_params['Pl_config']; Pr_start, Pr_end = pso_params['Pr_config']
    
    CONVERGENCE_THRESHOLD_GEN = pso_params['convergence_gen']
    TIME_LIMIT_SEC = pso_params['time_limit_sec']

    swarms = np.random.uniform(1, 20, (num_par, num_nodes)); velocities = np.zeros_like(swarms)
    pBests = np.copy(swarms); pBests_fitness = np.full(num_par, -float('inf'))
    gBest = swarms[0]; gBest_fitness = -float('inf')
    gBest_feasible_bn = -1; gBest_feasible_path = None
    
    stagnation_counter = 0
    last_best_bn = -1.0 
    terminated_reason = "max_gen"
    final_gen = num_gen 

    for i in range(num_gen): # 世代ループ
        # (タイムアウト判定、パラメータ更新は watase_26_gen_log.py と同一)
        current_pso_time = time.time() - pso_start_time
        if current_pso_time > TIME_LIMIT_SEC:
            if gBest_feasible_path is None: terminated_reason = "timeout_no_solution"
            else: terminated_reason = "timeout_with_solution"
            final_gen = i 
            break

        progress = i / num_gen
        w = w_start - (w_start - w_end) * progress; c1 = c1_start - (c1_start - c1_end) * progress
        c2 = c2_start + (c2_end - c2_start) * progress; P_d = Pd_start + (Pd_end - Pd_start) * progress
        P_l = Pl_start + (Pl_end - Pl_start) * progress; P_r = Pr_start + (Pr_end - Pr_start) * progress
        
        current_fitness = np.zeros(num_par)
        
        for j in range(num_par): # 粒子ループ
            path, is_valid = PathEncode(swarms[j], Graph, src, dst) # ★ 確率的なPathEncodeが呼ばれる
            
            # ▼▼▼ ★★★ 変更点: watase_26 (罰 -1.0) のロジックに戻す ★★★ ▼▼▼
            if not is_valid: 
                current_fitness[j] = -1.0 # 確率的選択が失敗したら罰を与える
                continue
            # ▲▲▲ ★★★ 変更ここまで ★★★ ▲▲▲

            # (is_valid=True の場合のみ、Fitness計算)
            bn, d, l, r = calculate_path_attributes_4d(Graph, path)
            fitness = bn; penalty = 0
            if d > max_delay: penalty += P_d * (d - max_delay)
            if l > max_loss: penalty += P_l * (l - max_loss)
            if r < min_rel: penalty += P_r * (min_rel - r)
            fitness -= penalty
            current_fitness[j] = fitness
            
            is_feasible = (d <= max_delay and l <= max_loss and r >= min_rel)
            
            if is_feasible and bn > gBest_feasible_bn:
                gBest_feasible_bn = bn 
                gBest_feasible_path = path

        # (pBest, gBest, 速度, 位置の更新は watase_26_gen_log.py と同一)
        update_indices = current_fitness > pBests_fitness
        pBests[update_indices] = swarms[update_indices]; pBests_fitness[update_indices] = current_fitness[update_indices]
        current_best_idx = np.argmax(current_fitness)
        if current_fitness[current_best_idx] > gBest_fitness:
            gBest_fitness = current_fitness[current_best_idx]; gBest = swarms[current_best_idx]
        
        r1, r2 = np.random.rand(2, num_par, 1)
        velocities = w * velocities + c1 * r1 * (pBests - swarms) + c2 * r2 * (gBest.reshape(1, -1) - swarms)
        swarms += velocities

        # (収束判定と世代ログ記録は watase_26_gen_log.py と同一)
        if gBest_feasible_bn > -1:
            if gBest_feasible_bn > last_best_bn:
                stagnation_counter = 0 
                last_best_bn = gBest_feasible_bn 
            else:
                stagnation_counter += 1 
            
            if stagnation_counter >= CONVERGENCE_THRESHOLD_GEN:
                terminated_reason = "convergence"
                final_gen = i + 1 
                pso_generation_history.append((i, time.time() - pso_start_time, gBest_feasible_bn))
                break 
        
        if terminated_reason != "convergence": 
            pso_generation_history.append((i, time.time() - pso_start_time, gBest_feasible_bn))
            
    # --- ループ終了後の処理 ---
    pso_time = time.time() - pso_start_time
    pso_bn, pso_d, pso_l, pso_r = calculate_path_attributes_4d(Graph, gBest_feasible_path) if gBest_feasible_path else (-1,-1,-1,-1)

    return (opt_bn, lc_time,
            pso_bn, pso_time, gBest_feasible_path is not None,
            pso_d, pso_l, pso_r,
            final_gen, terminated_reason,
            pso_generation_history)

if __name__ == '__main__':
    # (パラメータ設定は前回の「超・探索型」＋「正規化ペナルティ」を引き継ぐ)
    num_simulation = 30 
    node_counts = [1000]
    graph_types = ['random']
    constraints = {'delay_multiplier': 3.0, 'loss_constraint': 0.1, 'reliability_constraint': 0.95}
    
    pso_params = {
        'num_par': 100,
        'num_gen': 100, 
        'convergence_gen': 100,
        'time_limit_sec': 1200, 
        
        # 1. 「超・探索型」パラメータ
        'w_config': (0.9, 0.9),  
        'c1_config': (3.5, 3.0), 
        'c2_config': (0.5, 1.0),  

        # 2. 「正規化 (バランス)」ペナルティ
        'Pd_config': (0.1, 1.0), 
        'Pl_config': (100.0, 500.0),  
        'Pr_config': (500.0, 1000.0)
    }

    # (CSVの準備と実行ループは watase_26_gen_log.py と同一)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = savef.create_dir(dir_name="adaptive_termination_results_4D_FIXED")
    
    # ★★★ ファイル名に 'stochastic' を追加 ★★★
    generation_log_filename = result_path + f'/{run_timestamp}_4criteria_generation_log_STOCHASTIC.csv'
    summary_filename = result_path + f'/{run_timestamp}_4criteria_summary_STOCHASTIC.csv'
    
    # (CSVヘッダー書き込みは watase_26_gen_log.py と同一)
    try:
        with open(generation_log_filename, 'w', newline='') as f_log:
            log_writer = csv.writer(f_log)
            log_writer.writerow(['Num_Nodes', 'Graph_Type', 'Iter', 'Generation', 'Time', 'Bottleneck'])
    except IOError as e:
        print(f"時系列ログファイルの作成に失敗しました: {e}")
        exit()
        
    try:
        with open(summary_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Num_Nodes', 'Graph_Type', 'Iter', 'Optimal_BN', 'Optimal_Time',
                             'PSO_Final_BN', 'PSO_Final_Time', 'PSO_Feasible', 'PSO_Final_Delay', 'PSO_Final_Loss', 'PSO_Final_Reliability',
                             'PSO_Final_Gen', 'PSO_Termination_Reason'])
    except IOError as e:
        print(f"サマリーファイルの作成に失敗しました: {e}")
        exit()

    # (実行ループは watase_26_gen_log.py と同一)
    for num_nodes in node_counts:
        for graph_type in graph_types:
            for i in range(num_simulation):
                print(f'\n--- Sim (N:{num_nodes}, G:{graph_type}, iter:{i+1}) ---')
                Graph = getattr(create_graph, f"{graph_type}_graph")(num_node=num_nodes)
                if not nx.is_connected(Graph):
                    largest_cc = max(nx.connected_components(Graph), key=len)
                    if len(largest_cc) < 2: continue
                    node_list = list(largest_cc)
                else: 
                    node_list = list(Graph.nodes())
                if len(node_list) < 2: continue
                src, dst = random.sample(node_list, 2)

                (opt_bn, lc_time,
                 pso_bn, pso_time, pso_feasible, 
                 pso_d, pso_l, pso_r,
                 pso_final_gen, pso_term_reason,
                 pso_generation_history) = simulation(Graph, src, dst, constraints, pso_params)

                if opt_bn is None: 
                    continue

                try:
                    with open(generation_log_filename, 'a', newline='') as f_log:
                        log_writer = csv.writer(f_log)
                        for (gen, log_time, log_bn) in pso_generation_history:
                            log_writer.writerow([num_nodes, graph_type, i + 1, gen, log_time, log_bn])
                except IOError as e:
                    print(f"世代ログファイルへの書き込みエラー: {e}")

                try:
                    with open(summary_filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            num_nodes, graph_type, i + 1, opt_bn, lc_time,
                            pso_bn, pso_time, pso_feasible, pso_d, pso_l, pso_r,
                            pso_final_gen, pso_term_reason
                        ])
                except IOError as e:
                    print(f"サマリーファイルへの書き込みエラー: {e}")

    print(f"\n実験サマリーが '{summary_filename}' に保存されました。")
    print(f"世代ログが '{generation_log_filename}' に保存されました。")