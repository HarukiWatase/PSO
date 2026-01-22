# watase_42_singlecore_all.py
# 目的: 並列化なし(シングルコア)でのノード数別計算時間計測
# 制約数1 と 制約数3 の両方をループで実行する

import networkx as nx
import numpy as np
import create_graph_4 as create_graph
import savef
from datetime import datetime
import csv
import time
import math
import random
from heapq import heappop, heappush

# --- タイムアウト機能 (Signal使用はWindows非対応なことがあるため、ループ内チェックで実装) ---
EXACT_TIMEOUT = 600 # 10分以上かかったら諦める

# ==========================================
# 共通関数 (シングルスレッド用)
# ==========================================
def PathEncode(Particle, Graph, src, dst):
    path = [src]; current_node = src; visited = {src}; limit_len = len(Particle)
    while current_node != dst:
        neighbors = [n for n in Graph.neighbors(current_node) if n not in visited]
        if not neighbors: return path, False
        best_neighbor = -1; highest_prio = -1.0
        for neighbor in neighbors:
            if neighbor < limit_len:
                prio = Particle[neighbor]
                if prio > highest_prio: highest_prio = prio; best_neighbor = neighbor
        if best_neighbor == -1: return path, False
        current_node = best_neighbor
        path.append(current_node); visited.add(current_node)
    return path, True

def calculate_path_attributes_4d(G, path):
    if not path or len(path) < 2: return 0, float('inf'), 1.0, 0.0
    bottleneck = float('inf'); total_delay = 0; total_loss_log = 0; total_rel_cost = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge = G.edges[u, v]
        bottleneck = min(bottleneck, edge.get('weight', 0))
        total_delay += edge.get('delay', float('inf'))
        total_loss_log += edge.get('loss_log_cost', float('inf'))
        total_rel_cost += edge.get('reliability_cost', float('inf'))
    return bottleneck, total_delay, 1 - math.exp(-total_loss_log), math.exp(-total_rel_cost)

def get_spatial_lbest(swarms, pBests, pBests_fitness, k=5):
    sq_norms = np.sum(swarms**2, axis=1)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * np.dot(swarms, swarms.T)
    dist_sq = np.maximum(dist_sq, 0)
    nearest_indices = np.argpartition(dist_sq, kth=k, axis=1)[:, :k]
    lBest_matrix = np.zeros_like(swarms)
    for i in range(len(swarms)):
        neighbors = nearest_indices[i]
        best_local_idx = np.argmax(pBests_fitness[neighbors])
        lBest_matrix[i] = pBests[neighbors[best_local_idx]]
    return lBest_matrix

def get_global_lbest(swarms, gBest):
    return np.tile(gBest, (len(swarms), 1))

# --- 厳密解法 (汎用: 1制約 or 3制約) ---
def find_optimal_generic(G, source, target, constraints, mode_1const=False):
    # constraints: {'max_delay': val, 'max_loss_log': val, 'max_rel_cost': val}
    labels = {node: [] for node in G.nodes()}
    pq = []
    
    # Label: (delay, neg_bn, loss_log, rel_cost)
    initial_label = (0.0, float('inf'), 0.0, 0.0) 
    labels[source].append(initial_label)
    heappush(pq, (-initial_label[1], initial_label[0], initial_label[2], initial_label[3], source, initial_label))

    best_bottle = -1
    start_time = time.time()

    while pq:
        # タイムアウトチェック
        if time.time() - start_time > EXACT_TIMEOUT:
            return -2 # Timeout

        neg_bottle, d_curr, l_curr, r_curr, u, label_curr = heappop(pq)
        if label_curr not in labels[u]: continue
        
        for v in G.neighbors(u):
            edge = G.get_edge_data(u, v)
            d_new = d_curr + edge.get("delay", 0)
            b_new = min(-neg_bottle, edge.get("weight", 1))
            
            l_new = l_curr + edge.get("loss_log_cost", 0)
            r_new = r_curr + edge.get("reliability_cost", 0)
            
            # 制約チェック
            if d_new > constraints['max_delay']: continue
            if not mode_1const:
                if l_new > constraints['max_loss_log']: continue
                if r_new > constraints['max_rel_cost']: continue
            
            label_new = (d_new, b_new, l_new, r_new)
            
            # ドミナンスチェック
            is_dominated = False
            for d, b, l, r in labels[v]:
                # 3制約モード
                if not mode_1const:
                    if d <= d_new and b >= b_new and l <= l_new and r <= r_new:
                        is_dominated = True; break
                # 1制約モード(Delayのみ)
                else:
                    if d <= d_new and b >= b_new:
                        is_dominated = True; break
            if is_dominated: continue
            
            # 劣るラベルの削除
            if not mode_1const:
                labels[v] = [lbl for lbl in labels[v] if not (lbl[0] >= d_new and lbl[1] <= b_new and lbl[2] >= l_new and lbl[3] >= r_new)]
            else:
                labels[v] = [lbl for lbl in labels[v] if not (lbl[0] >= d_new and lbl[1] <= b_new)]
                
            labels[v].append(label_new)
            heappush(pq, (-b_new, d_new, l_new, r_new, v, label_new))

    final_labels = labels.get(target, [])
    for d, b, l, r in final_labels:
        if b > best_bottle: best_bottle = b
    return best_bottle

# ==========================================
# PSO (シングルスレッド・汎用)
# ==========================================
def run_pso_single(Graph, src, dst, constraints, pso_params, mode_1const, topology='spatial', enable_restart=True):
    # 絶対値制約計算
    try:
        min_d_path = nx.dijkstra_path(Graph, source=src, target=dst, weight='delay')
        _, min_d, _, _ = calculate_path_attributes_4d(Graph, min_d_path)
        abs_max_delay = min_d * constraints['delay_multiplier']
    except: return -1, 0

    num_nodes = len(Graph.nodes()); num_par = pso_params['num_par']; num_gen = pso_params['num_gen']
    w_start, w_end = pso_params['w_config']
    c1_start, c1_end = pso_params['c1_config']; c2_start, c2_end = pso_params['c2_config']
    Pd_start, Pd_end = pso_params['Pd_config']
    
    if mode_1const:
        Pl_start = Pl_end = 0; Pr_start = Pr_end = 0
    else:
        Pl_start, Pl_end = pso_params['Pl_config']
        Pr_start, Pr_end = pso_params['Pr_config']
        
    start_time = time.time()
    
    swarms = np.random.uniform(1, 20, (num_par, num_nodes))
    velocities = np.zeros_like(swarms)
    pBests = np.copy(swarms); pBests_fitness = np.full(num_par, -float('inf'))
    gBest = swarms[0]; gBest_fitness = -float('inf'); gBest_feasible_bn = -1
    last_best_bn = -1.0; restart_counter = 0

    for i in range(num_gen):
        if enable_restart:
            if gBest_feasible_bn > last_best_bn:
                restart_counter = 0; last_best_bn = gBest_feasible_bn
            else: restart_counter += 1
            if restart_counter >= 20:
                swarms = np.random.uniform(1, 20, (num_par, num_nodes)); velocities = np.zeros_like(swarms)
                if gBest is not None: swarms[0] = gBest
                pBests = np.copy(swarms); pBests_fitness = np.full(num_par, -float('inf')); restart_counter = 0

        progress = i / num_gen
        w = w_start - (w_start - w_end) * progress
        c1 = c1_start - (c1_start - c1_end) * progress; c2 = c2_start + (c2_end - c2_start) * progress
        P_d = Pd_start + (Pd_end - Pd_start) * progress
        P_l = Pl_start + (Pl_end - Pl_start) * progress; P_r = Pr_start + (Pr_end - Pr_start) * progress
        
        current_fitness = np.zeros(num_par)
        
        for j in range(num_par):
            path, is_valid = PathEncode(swarms[j], Graph, src, dst)
            if not is_valid: 
                current_fitness[j] = -1.0; continue
            
            bn, d, l, r = calculate_path_attributes_4d(Graph, path)
            penalty = 0
            if d > abs_max_delay: penalty += P_d * (d - abs_max_delay)
            
            is_feasible = (d <= abs_max_delay)
            if not mode_1const:
                if l > constraints['loss_constraint']: penalty += P_l * (l - constraints['loss_constraint'])
                if r < constraints['reliability_constraint']: penalty += P_r * (constraints['reliability_constraint'] - r)
                is_feasible = is_feasible and (l <= constraints['loss_constraint'] and r >= constraints['reliability_constraint'])
            
            fitness = bn - penalty
            current_fitness[j] = fitness
            if is_feasible and bn > gBest_feasible_bn: gBest_feasible_bn = bn

        update = current_fitness > pBests_fitness
        pBests[update] = swarms[update]; pBests_fitness[update] = current_fitness[update]
        curr_best_idx = np.argmax(current_fitness)
        if current_fitness[curr_best_idx] > gBest_fitness:
            gBest_fitness = current_fitness[curr_best_idx]; gBest = swarms[curr_best_idx]

        if topology == 'spatial': lBest_matrix = get_spatial_lbest(swarms, pBests, pBests_fitness)
        else: lBest_matrix = get_global_lbest(swarms, gBest)
        
        r1, r2 = np.random.rand(2, num_par, 1)
        velocities = w * velocities + c1 * r1 * (pBests - swarms) + c2 * r2 * (lBest_matrix - swarms)
        swarms += velocities

    return gBest_feasible_bn, time.time() - start_time

# ==========================================
# Main
# ==========================================
if __name__ == '__main__':
    node_counts = [100, 500, 1000, 2000, 3000, 4000]
    num_trials = 5
    
    # シナリオ設定: 1=DelayOnly, 3=AllConstraints
    scenarios = [
        {'id': '1Const', 'mode_1const': True},
        {'id': '3Const', 'mode_1const': False}
    ]
    
    methods = [
        {'name': 'Global', 'topo': 'global', 'restart': False},
        {'name': 'Restart', 'topo': 'spatial', 'restart': True}
    ]
    
    pso_params = {
        'num_par': 200, 'num_gen': 200, 
        'w_config': (0.9, 0.18), 'c1_config': (3.32, 0.44), 'c2_config': (0.88,3.66),
        'Pd_config': (6.63, 56.4), 
        'Pl_config': (73.9, 56447.8),  
        'Pr_config': (73.9, 56447.8)
    }
    base_constraints = {'delay_multiplier': 3.0, 'loss_constraint': 0.1, 'reliability_constraint': 0.95}

    result_path = savef.create_dir(dir_name="singlecore_benchmark")
    csv_file = f"{result_path}/singlecore_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    print(f"=== [並列化なし] 全データ収集 (制約1/制約3) ===")
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Scenario', 'NodeCount', 'Trial', 'Method', 'BN', 'Time'])
        
        for n in node_counts:
            for t in range(num_trials):
                print(f"\n--- Node: {n}, Trial: {t+1} ---")
                Graph = create_graph.random_graph(num_node=n)
                if not nx.is_connected(Graph):
                    largest_cc = max(nx.connected_components(Graph), key=len)
                    nodes = list(largest_cc)
                else: nodes = list(Graph.nodes())
                if len(nodes) < 2: continue
                src, dst = random.sample(nodes, 2)
                
                try:
                    min_d_path = nx.dijkstra_path(Graph, source=src, target=dst, weight='delay')
                    _, min_d, _, _ = calculate_path_attributes_4d(Graph, min_d_path)
                    abs_max_delay = min_d * base_constraints['delay_multiplier']
                except: continue # パスなしスキップ

                # シナリオごとに実行
                for sc in scenarios:
                    s_id = sc['id']
                    is_1const = sc['mode_1const']
                    print(f"  [{s_id}] Executing...", end="\r")
                    
                    # 1. 厳密解法
                    print(f"  [{s_id}] Exact...", end="\r")
                    exact_const = {
                        'max_delay': abs_max_delay,
                        'max_loss_log': -math.log(1 - base_constraints['loss_constraint']),
                        'max_rel_cost': -math.log(base_constraints['reliability_constraint'])
                    }
                    
                    st = time.time()
                    exact_bn = find_optimal_generic(Graph, src, dst, exact_const, mode_1const=is_1const)
                    et = time.time() - st
                    
                    if exact_bn == -2: # Timeout
                        print(f"  [{s_id}] Exact TIMEOUT!      ")
                        writer.writerow([s_id, n, t+1, 'Exact', -1, -1]) # -1 time means timeout
                    else:
                        writer.writerow([s_id, n, t+1, 'Exact', exact_bn, et])
                        
                    # 2. PSO手法
                    for m in methods:
                        print(f"  [{s_id}] {m['name']}...", end="\r")
                        bn, exec_time = run_pso_single(
                            Graph, src, dst, base_constraints, pso_params, 
                            mode_1const=is_1const, topology=m['topo'], enable_restart=m['restart']
                        )
                        writer.writerow([s_id, n, t+1, m['name'], bn, exec_time])
                    
                    f.flush()
    print(f"\n全実験終了。結果: {csv_file}")