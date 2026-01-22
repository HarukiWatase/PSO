# watase_41_parallel_1const.py
# 目的: 並列化あり・制約数1(遅延のみ)における計算時間計測

import networkx as nx
import numpy as np
import create_graph_4 as create_graph
import savef
from datetime import datetime
import csv
import time
import math
import random
import multiprocessing
from heapq import heappop, heappush

# ==========================================
# 1. 並列化用ワーカー
# ==========================================
worker_graph = None; worker_src = None; worker_dst = None; worker_constraints = None

def init_worker(G, s, d, const):
    global worker_graph, worker_src, worker_dst, worker_constraints
    worker_graph = G; worker_src = s; worker_dst = d; worker_constraints = const

def fitness_worker(args):
    particle, P_d, P_l, P_r = args # P_l, P_r は 0 が渡される想定
    path, is_valid = PathEncode(particle, worker_graph, worker_src, worker_dst)
    
    if not is_valid: return -1.0, False, -1.0
    
    bn, d, l, r = calculate_path_attributes_4d(worker_graph, path)
    
    max_delay = worker_constraints['delay_multiplier']
    
    # 制約数1なので、遅延以外のペナルティは無視（係数が0になるため計算上は残しても良いが明示的に）
    penalty = 0
    if d > max_delay: penalty += P_d * (d - max_delay)
    # Loss, Reliabilityは無視
    
    fitness = bn - penalty
    # 実行可能判定も遅延のみ
    is_feasible = (d <= max_delay)
    
    return fitness, is_feasible, bn

# ==========================================
# 2. 共通関数
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

# --- 厳密解法 (1制約用: 遅延制約付きボトルネック最大化) ---
# 実質的に、エッジをボトルネック順にソートしてフィルタリングする等の高速法があるが、
# 比較のためラベル修正法ベースで行く（ただし他のコストは無視）
def find_optimal_1const(G, source, target, max_delay):
    labels = {node: [] for node in G.nodes()}
    pq = []
    # (delay, neg_bn)
    initial_label = (0.0, float('inf')) 
    labels[source].append(initial_label)
    heappush(pq, (-initial_label[1], initial_label[0], source, initial_label))

    best_bottle = -1

    while pq:
        neg_bottle, d_curr, u, label_curr = heappop(pq)
        if label_curr not in labels[u]: continue
        
        for v in G.neighbors(u):
            edge = G.get_edge_data(u, v)
            d_new = d_curr + edge.get("delay", 0)
            b_new = min(-neg_bottle, edge.get("weight", 1))
            
            if d_new > max_delay: continue
            
            label_new = (d_new, b_new)
            is_dominated = False
            for d, b in labels[v]:
                # 遅延が大きく、かつ帯域も小さいなら支配される
                if d <= d_new and b >= b_new:
                    is_dominated = True; break
            if is_dominated: continue
            
            labels[v] = [lbl for lbl in labels[v] if not (lbl[0] >= d_new and lbl[1] <= b_new)]
            labels[v].append(label_new)
            heappush(pq, (-b_new, d_new, v, label_new))

    final_labels = labels.get(target, [])
    for d, b in final_labels:
        if b > best_bottle: best_bottle = b
    return best_bottle

# ==========================================
# 3. 並列PSO (1制約版)
# ==========================================
def run_pso_parallel_1const(Graph, src, dst, constraints, pso_params, num_cores, topology='spatial', enable_restart=True):
    try:
        min_delay_path = nx.dijkstra_path(Graph, source=src, target=dst, weight='delay')
        _, min_delay, _, _ = calculate_path_attributes_4d(Graph, min_delay_path)
        abs_max_delay = min_delay * constraints['delay_multiplier']
        constraints_for_worker = constraints.copy()
        constraints_for_worker['delay_multiplier'] = abs_max_delay
    except:
        return -1.0, 0.0

    num_nodes = len(Graph.nodes()); num_par = pso_params['num_par']; num_gen = pso_params['num_gen']
    w_start, w_end = pso_params['w_config']
    c1_start, c1_end = pso_params['c1_config']; c2_start, c2_end = pso_params['c2_config']
    Pd_start, Pd_end = pso_params['Pd_config']
    
    # 1制約なので P_l, P_r は 0 に固定
    Pl_start = Pl_end = 0
    Pr_start = Pr_end = 0
    
    pso_start_time = time.time()
    
    RESTART_THRESHOLD = 20; restart_counter = 0

    swarms = np.random.uniform(1, 20, (num_par, num_nodes))
    velocities = np.zeros_like(swarms)
    pBests = np.copy(swarms); pBests_fitness = np.full(num_par, -float('inf'))
    gBest = swarms[0]; gBest_fitness = -float('inf')
    gBest_feasible_bn = -1; last_best_bn = -1.0 

    with multiprocessing.Pool(processes=num_cores, initializer=init_worker, initargs=(Graph, src, dst, constraints_for_worker)) as pool:
        for i in range(num_gen):
            if enable_restart:
                if gBest_feasible_bn > last_best_bn:
                    restart_counter = 0; last_best_bn = gBest_feasible_bn
                else: restart_counter += 1
                if restart_counter >= RESTART_THRESHOLD:
                    swarms = np.random.uniform(1, 20, (num_par, num_nodes))
                    velocities = np.zeros_like(swarms)
                    if gBest is not None: swarms[0] = gBest 
                    pBests = np.copy(swarms); pBests_fitness = np.full(num_par, -float('inf'))
                    restart_counter = 0

            progress = i / num_gen
            w = w_start - (w_start - w_end) * progress
            c1 = c1_start - (c1_start - c1_end) * progress; c2 = c2_start + (c2_end - c2_start) * progress
            P_d = Pd_start + (Pd_end - Pd_start) * progress
            P_l = 0; P_r = 0 # ゼロ固定
            
            args_list = [(swarms[j], P_d, P_l, P_r) for j in range(num_par)]
            results = pool.map(fitness_worker, args_list)
            
            current_fitness = np.zeros(num_par)
            for j, (fit, is_feas, bn) in enumerate(results):
                current_fitness[j] = fit
                if is_feas and bn > gBest_feasible_bn:
                    gBest_feasible_bn = bn

            update_indices = current_fitness > pBests_fitness
            pBests[update_indices] = swarms[update_indices]; pBests_fitness[update_indices] = current_fitness[update_indices]
            current_best_idx = np.argmax(current_fitness)
            if current_fitness[current_best_idx] > gBest_fitness:
                gBest_fitness = current_fitness[current_best_idx]; gBest = swarms[current_best_idx]
            
            if topology == 'spatial': lBest_matrix = get_spatial_lbest(swarms, pBests, pBests_fitness, k=5)
            else: lBest_matrix = get_global_lbest(swarms, gBest)

            r1 = np.random.rand(num_par, 1); r2 = np.random.rand(num_par, 1)
            velocities = w * velocities + c1 * r1 * (pBests - swarms) + c2 * r2 * (lBest_matrix - swarms)
            swarms += velocities

    total_time = time.time() - pso_start_time
    return gBest_feasible_bn, total_time

# ==========================================
# 4. Main
# ==========================================
if __name__ == '__main__':
    node_counts = [100, 500, 1000, 2000, 3000, 4000] # 要件に合わせて設定
    num_trials = 5
    num_cores = 8 
    
    # 手法: Exact(Dijkstraベース), Global, Restart
    methods = [
        {'name': 'Global', 'topo': 'global', 'restart': False},
        {'name': 'Restart', 'topo': 'spatial', 'restart': True}
    ]
    
    pso_params = {
        'num_par': 200, 'num_gen': 200, 
        'w_config': (0.9, 0.18), 'c1_config': (3.32, 0.44), 'c2_config': (0.88,3.66),
        'Pd_config': (6.63, 56.4) # Delay Penaltyのみ
    }
    constraints = {'delay_multiplier': 3.0}

    result_path = savef.create_dir(dir_name="parallel_1const_benchmark")
    csv_file = f"{result_path}/parallel_1const_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    print(f"=== [並列化/1制約] ノード数別計算時間計測 ===")

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['NodeCount', 'Trial', 'Method', 'BN', 'Time'])
        
        for n in node_counts:
            print(f"\n>>> Node Count: {n} <<<")
            
            for t in range(num_trials):
                print(f"  Trial {t+1}/{num_trials} generating...", end="\r")
                Graph = create_graph.random_graph(num_node=n)
                if not nx.is_connected(Graph):
                    largest_cc = max(nx.connected_components(Graph), key=len)
                    nodes = list(largest_cc)
                else: nodes = list(Graph.nodes())
                if len(nodes) < 2: continue
                src, dst = random.sample(nodes, 2)
                
                # --- 厳密解法 (1制約) ---
                print(f"  Trial {t+1} [Exact]...", end="\r")
                try:
                    min_d_path = nx.dijkstra_path(Graph, source=src, target=dst, weight='delay')
                    _, min_d, _, _ = calculate_path_attributes_4d(Graph, min_d_path)
                    abs_max_delay = min_d * constraints['delay_multiplier']
                    
                    st = time.time()
                    exact_bn = find_optimal_1const(Graph, src, dst, abs_max_delay)
                    et = time.time() - st
                    writer.writerow([n, t+1, 'Exact', exact_bn, et])
                except:
                    writer.writerow([n, t+1, 'Exact', -1, 0])
                
                # --- PSO手法 ---
                for m in methods:
                    print(f"  Trial {t+1} [{m['name']}]...", end="\r")
                    bn, exec_time = run_pso_parallel_1const(
                        Graph, src, dst, constraints, pso_params, 
                        num_cores=num_cores, topology=m['topo'], enable_restart=m['restart']
                    )
                    writer.writerow([n, t+1, m['name'], bn, exec_time])
                
                f.flush()

    print(f"\n実験終了。結果: {csv_file}")