# watase_31.py
# 構成: 空間的近傍(Step1) + 高速化PathEncode + 論理的ペナルティ + 停滞時Restart

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

# --- 1. 高速化されたPathEncode (リスト内包表記) ---
def PathEncode(Particle, Graph, src, dst):
    path = [src]
    current_node = src
    visited = {src}
    
    # 粒子（優先度リスト）のサイズを事前に取得
    limit_len = len(Particle)
    
    while current_node != dst:
        # Pythonのforループは遅いため、filter処理をC言語レベル(リスト内包表記)で行う
        neighbors = [n for n in Graph.neighbors(current_node) if n not in visited]
        
        if not neighbors: 
            return path, False
            
        best_neighbor = -1
        highest_prio = -1.0
        
        # 絞り込まれた少数の要素に対するループなので、ここはPythonでも十分速い
        for neighbor in neighbors:
            if neighbor < limit_len:
                prio = Particle[neighbor]
                if prio > highest_prio:
                    highest_prio = prio
                    best_neighbor = neighbor
        
        if best_neighbor == -1:
            return path, False
            
        current_node = best_neighbor
        path.append(current_node)
        visited.add(current_node)
        
    return path, True

# --- 属性計算 (変更なし) ---
def calculate_path_attributes_4d(G, path):
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

# --- 厳密解法 (変更なし) ---
def find_optimal_path_by_label_correcting_4d(G, source, target, max_delay, max_loss_rate, min_reliability):
    labels = {node: [] for node in G.nodes()}; pred = {node: {} for node in G.nodes()}; pq = []
    initial_label = (0.0, float('inf'), 0.0, 0.0) 
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

# --- 空間的近傍 (Step 1: ユークリッド距離) ---
# ※Step 2 (パス類似度) は計算コストと精度の問題で除外
def get_spatial_lbest(swarms, pBests, pBests_fitness, k=5):
    """
    空間的近傍（ユークリッド距離）に基づいて各粒子のlBestを決定する。
    """
    num_par = swarms.shape[0]
    
    # 1. 全粒子間のユークリッド距離の二乗を高速計算
    sq_norms = np.sum(swarms**2, axis=1)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * np.dot(swarms, swarms.T)
    dist_sq = np.maximum(dist_sq, 0)
    
    # 2. 各粒子ごとに距離が近い上位k個のインデックスを取得
    nearest_indices = np.argpartition(dist_sq, kth=k, axis=1)[:, :k]
    
    # 3. 近傍k個の中で最高のpBestを持つインデックスを選択
    lBest_matrix = np.zeros_like(swarms)
    
    for i in range(num_par):
        neighbors = nearest_indices[i]
        neighbor_fitness = pBests_fitness[neighbors]
        best_local_idx = np.argmax(neighbor_fitness)
        best_global_idx = neighbors[best_local_idx]
        lBest_matrix[i] = pBests[best_global_idx]
        
    return lBest_matrix

# --- Simulation (Restart機能追加) ---
def simulation(Graph, src, dst, constraints, pso_params):
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

    # --- PSO ---
    pso_start_time = time.time()
    pso_generation_history = [] 

    num_nodes, num_par, num_gen = len(Graph.nodes()), pso_params['num_par'], pso_params['num_gen']
    w_start, w_end = pso_params['w_config']; c1_start, c1_end = pso_params['c1_config']
    c2_start, c2_end = pso_params['c2_config']; Pd_start, Pd_end = pso_params['Pd_config']
    Pl_start, Pl_end = pso_params['Pl_config']; Pr_start, Pr_end = pso_params['Pr_config']
    
    CONVERGENCE_THRESHOLD_GEN = pso_params['convergence_gen']
    TIME_LIMIT_SEC = pso_params['time_limit_sec']
    
    # ★Restart用パラメータ
    RESTART_THRESHOLD = 20  # 20世代改善がなければ爆発
    restart_counter = 0

    swarms = np.random.uniform(1, 20, (num_par, num_nodes)); velocities = np.zeros_like(swarms)
    pBests = np.copy(swarms); pBests_fitness = np.full(num_par, -float('inf'))
    gBest = swarms[0]; gBest_fitness = -float('inf')
    gBest_feasible_bn = -1; gBest_feasible_path = None
    
    stagnation_counter = 0; last_best_bn = -1.0; terminated_reason = "max_gen"; final_gen = num_gen 

    for i in range(num_gen):
        current_pso_time = time.time() - pso_start_time
        if current_pso_time > TIME_LIMIT_SEC:
            terminated_reason = "timeout" if gBest_feasible_path is None else "timeout_with_solution"
            final_gen = i 
            break

        # --- ★停滞検知と再初期化 (Restart Logic) ---
        if gBest_feasible_bn > last_best_bn:
            restart_counter = 0
            stagnation_counter = 0 # 収束判定用のカウンタもリセット
            last_best_bn = gBest_feasible_bn
        else:
            restart_counter += 1
            stagnation_counter += 1
            
        # Restartの発動
        if restart_counter >= RESTART_THRESHOLD:
            # print(f"Gen {i}: Explosion triggered! (Current Best: {gBest_feasible_bn})") # デバッグ用
            
            # 1. 全員をランダム再配置
            swarms = np.random.uniform(1, 20, (num_par, num_nodes))
            velocities = np.zeros_like(swarms)
            
            # 2. エリート保存 (ID:0 に現在のgBestを強制代入)
            if gBest is not None:
                swarms[0] = gBest
                
            # 3. 記憶のリセット (pBestを現在位置に、Fitnessをリセット)
            # これにより粒子は過去の局所解を忘れ、新しい場所から探索を始める
            pBests = np.copy(swarms)
            pBests_fitness = np.full(num_par, -float('inf'))
            
            restart_counter = 0
        # ----------------------------------------

        progress = i / num_gen
        w = w_start - (w_start - w_end) * progress; c1 = c1_start - (c1_start - c1_end) * progress
        c2 = c2_start + (c2_end - c2_start) * progress; P_d = Pd_start + (Pd_end - Pd_start) * progress
        P_l = Pl_start + (Pl_end - Pl_start) * progress; P_r = Pr_start + (Pr_end - Pr_start) * progress
        
        current_fitness = np.zeros(num_par)
        
        for j in range(num_par):
            path, is_valid = PathEncode(swarms[j], Graph, src, dst)
            
            if not is_valid: 
                current_fitness[j] = -1.0
                continue
            
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

        update_indices = current_fitness > pBests_fitness
        pBests[update_indices] = swarms[update_indices]; pBests_fitness[update_indices] = current_fitness[update_indices]
        
        current_best_idx = np.argmax(current_fitness)
        if current_fitness[current_best_idx] > gBest_fitness:
            gBest_fitness = current_fitness[current_best_idx]; gBest = swarms[current_best_idx]
        
        # --- 近傍計算 (Step 1: 空間的近傍) ---
        # k=5 (標準設定)
        lBest_matrix = get_spatial_lbest(swarms, pBests, pBests_fitness, k=5)
        
        r1, r2 = np.random.rand(2, num_par, 1)
        velocities = w * velocities + c1 * r1 * (pBests - swarms) + c2 * r2 * (lBest_matrix - swarms)
        swarms += velocities

        # 収束判定 (Restartがあるため、閾値を長めにするか、Restart回数制限などを入れる場合もあるが、今回はそのまま)
        # ただしRestartが発動するとstagnation_counterもリセットされる仕様にしているので、
        # Restartしてもなお改善しない（真の限界）場合のみ停止する挙動になる
        if stagnation_counter >= CONVERGENCE_THRESHOLD_GEN:
            terminated_reason = "convergence"; final_gen = i + 1 
            pso_generation_history.append((i, time.time() - pso_start_time, gBest_feasible_bn))
            break 
        
        if terminated_reason != "convergence":
            pso_generation_history.append((i, time.time() - pso_start_time, gBest_feasible_bn))
            
    pso_time = time.time() - pso_start_time
    pso_bn, pso_d, pso_l, pso_r = calculate_path_attributes_4d(Graph, gBest_feasible_path) if gBest_feasible_path else (-1,-1,-1,-1)

    return (opt_bn, lc_time, pso_bn, pso_time, gBest_feasible_path is not None, pso_d, pso_l, pso_r, final_gen, terminated_reason, pso_generation_history)

if __name__ == '__main__':
    num_simulation = 20 
    node_counts = [3000] 
    graph_types = ['random']
    constraints = {'delay_multiplier': 3.0, 'loss_constraint': 0.1, 'reliability_constraint': 0.95}
    
    # --- パラメータ設定 (論理的ペナルティ + 成功した速度設定) ---
    pso_params = {
        'num_par': 150, 'num_gen': 100,
        'convergence_gen': 100,
        'time_limit_sec': 1200, 
        'w_config': (0.9, 0.18), 'c1_config': (3.32, 0.44), 'c2_config': (0.88,3.66),
        # 2. ペナルティ
        # Delay: 序盤は0.1(無視) -> 終盤は100.0(絶対遵守)
        'Pd_config': (6.63, 56.4), 
        'Pl_config': (73.9, 56447.8),  
        'Pr_config': (73.9, 56447.8)
    }


    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = savef.create_dir(dir_name="adaptive_results_Restart")
    
    generation_log_filename = result_path + f'/{run_timestamp}_restart_generation_log.csv'
    try:
        with open(generation_log_filename, 'w', newline='') as f_log:
            log_writer = csv.writer(f_log)
            log_writer.writerow(['Num_Nodes', 'Graph_Type', 'Iter', 'Generation', 'Time', 'Bottleneck'])
    except IOError as e: print(e); exit()
        
    summary_filename = result_path + f'/{run_timestamp}_restart_summary.csv'
    try:
        with open(summary_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Num_Nodes', 'Graph_Type', 'Iter', 'Optimal_BN', 'Optimal_Time',
                             'PSO_Final_BN', 'PSO_Final_Time', 'PSO_Feasible', 'PSO_Final_Delay', 'PSO_Final_Loss', 'PSO_Final_Reliability',
                             'PSO_Final_Gen', 'PSO_Termination_Reason'])
    except IOError as e: print(e); exit()

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

                (opt_bn, lc_time, pso_bn, pso_time, pso_feasible, pso_d, pso_l, pso_r, pso_final_gen, pso_term_reason, pso_generation_history) = simulation(Graph, src, dst, constraints, pso_params)

                if opt_bn is None: continue

                try:
                    with open(generation_log_filename, 'a', newline='') as f_log:
                        log_writer = csv.writer(f_log)
                        for (gen, log_time, log_bn) in pso_generation_history:
                            log_writer.writerow([num_nodes, graph_type, i + 1, gen, log_time, log_bn])
                except IOError: pass

                try:
                    with open(summary_filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([num_nodes, graph_type, i + 1, opt_bn, lc_time, pso_bn, pso_time, pso_feasible, pso_d, pso_l, pso_r, pso_final_gen, pso_term_reason])
                except IOError: pass

    print(f"\n実験サマリーが '{summary_filename}' に保存されました。")
    print(f"世代ログが '{generation_log_filename}' に保存されました。")