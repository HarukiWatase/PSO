# watase_37.py
# 目的: 並列化スケーラビリティの計測（コア数 × 3手法）
# 同じグラフを用いて公平に比較する

import networkx as nx
import numpy as np
import create_graph_4 as create_graph
import savef
from datetime import datetime
import csv
import time
import math
import random
import multiprocessing # 並列化用

# ==========================================
# 1. 並列化用ワーカー設定 (Global Scope)
# ==========================================

worker_graph = None
worker_src = None
worker_dst = None
worker_constraints = None

def init_worker(G, s, d, const):
    global worker_graph, worker_src, worker_dst, worker_constraints
    worker_graph = G
    worker_src = s
    worker_dst = d
    worker_constraints = const

def fitness_worker(args):
    particle, P_d, P_l, P_r = args
    path, is_valid = PathEncode(particle, worker_graph, worker_src, worker_dst)
    
    if not is_valid:
        return -1.0, False, -1.0
    
    bn, d, l, r = calculate_path_attributes_4d(worker_graph, path)
    
    max_delay = worker_constraints['delay_multiplier']
    max_loss = worker_constraints['loss_constraint']
    min_rel = worker_constraints['reliability_constraint']
    
    penalty = 0
    if d > max_delay: penalty += P_d * (d - max_delay)
    if l > max_loss: penalty += P_l * (l - max_loss)
    if r < min_rel: penalty += P_r * (min_rel - r)
    
    fitness = bn - penalty
    is_feasible = (d <= max_delay and l <= max_loss and r >= min_rel)
    return fitness, is_feasible, bn

# ==========================================
# 2. 共通関数群
# ==========================================

def PathEncode(Particle, Graph, src, dst):
    path = [src]
    current_node = src
    visited = {src}
    limit_len = len(Particle)
    
    while current_node != dst:
        neighbors = [n for n in Graph.neighbors(current_node) if n not in visited]
        if not neighbors: return path, False
        
        best_neighbor = -1
        highest_prio = -1.0
        
        for neighbor in neighbors:
            if neighbor < limit_len:
                prio = Particle[neighbor]
                if prio > highest_prio: 
                    highest_prio = prio
                    best_neighbor = neighbor
                    
        if best_neighbor == -1: return path, False
        
        current_node = best_neighbor
        path.append(current_node)
        visited.add(current_node)
        
    return path, True

def calculate_path_attributes_4d(G, path):
    if not path or len(path) < 2: return 0, float('inf'), 1.0, 0.0
    bottleneck = float('inf')
    total_delay = 0
    total_loss_log_cost = 0
    total_reliability_cost = 0
    
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

# ==========================================
# 3. 並列化 PSO (手法切替対応版)
# ==========================================

def run_pso_parallel_scalability(Graph, src, dst, constraints, pso_params, num_cores, topology='spatial', enable_restart=True):
    # 事前計算
    try:
        min_delay_path = nx.dijkstra_path(Graph, source=src, target=dst, weight='delay')
        _, min_delay, _, _ = calculate_path_attributes_4d(Graph, min_delay_path)
        abs_max_delay = min_delay * constraints['delay_multiplier']
        constraints_for_worker = constraints.copy()
        constraints_for_worker['delay_multiplier'] = abs_max_delay
    except:
        return -1.0, 0.0

    num_nodes = len(Graph.nodes())
    num_par = pso_params['num_par']
    num_gen = pso_params['num_gen']
    
    # Params
    w_start, w_end = pso_params['w_config']
    c1_start, c1_end = pso_params['c1_config']
    c2_start, c2_end = pso_params['c2_config']
    Pd_start, Pd_end = pso_params['Pd_config']
    Pl_start, Pl_end = pso_params['Pl_config']
    Pr_start, Pr_end = pso_params['Pr_config']
    
    pso_start_time = time.time()
    
    RESTART_THRESHOLD = 20
    restart_counter = 0

    # Initialize
    swarms = np.random.uniform(1, 20, (num_par, num_nodes))
    velocities = np.zeros_like(swarms)
    pBests = np.copy(swarms)
    pBests_fitness = np.full(num_par, -float('inf'))
    gBest = swarms[0]
    gBest_fitness = -float('inf')
    gBest_feasible_bn = -1
    last_best_bn = -1.0 

    with multiprocessing.Pool(
        processes=num_cores,
        initializer=init_worker,
        initargs=(Graph, src, dst, constraints_for_worker)
    ) as pool:

        for i in range(num_gen):
            # Restart Logic
            if enable_restart:
                if gBest_feasible_bn > last_best_bn:
                    restart_counter = 0
                    last_best_bn = gBest_feasible_bn
                else:
                    restart_counter += 1
                
                if restart_counter >= RESTART_THRESHOLD:
                    # Explosion
                    swarms = np.random.uniform(1, 20, (num_par, num_nodes))
                    velocities = np.zeros_like(swarms)
                    if gBest is not None: swarms[0] = gBest # Keep Elite
                    pBests = np.copy(swarms)
                    pBests_fitness = np.full(num_par, -float('inf'))
                    restart_counter = 0

            # Dynamic Parameters
            progress = i / num_gen
            w = w_start - (w_start - w_end) * progress
            c1 = c1_start - (c1_start - c1_end) * progress
            c2 = c2_start + (c2_end - c2_start) * progress
            P_d = Pd_start + (Pd_end - Pd_start) * progress
            P_l = Pl_start + (Pl_end - Pl_start) * progress
            P_r = Pr_start + (Pr_end - Pr_start) * progress
            
            # Parallel Evaluation
            args_list = [(swarms[j], P_d, P_l, P_r) for j in range(num_par)]
            results = pool.map(fitness_worker, args_list)
            
            # Aggregate Results
            current_fitness = np.zeros(num_par)
            for j, (fit, is_feas, bn) in enumerate(results):
                current_fitness[j] = fit
                if is_feas and bn > gBest_feasible_bn:
                    gBest_feasible_bn = bn

            # Update pBest / gBest
            update_indices = current_fitness > pBests_fitness
            pBests[update_indices] = swarms[update_indices]
            pBests_fitness[update_indices] = current_fitness[update_indices]
            
            current_best_idx = np.argmax(current_fitness)
            if current_fitness[current_best_idx] > gBest_fitness:
                gBest_fitness = current_fitness[current_best_idx]
                gBest = swarms[current_best_idx]
            
            # Topology Selection
            if topology == 'spatial':
                lBest_matrix = get_spatial_lbest(swarms, pBests, pBests_fitness, k=5)
            else: # global
                lBest_matrix = get_global_lbest(swarms, gBest)

            # Update Velocity/Position
            r1 = np.random.rand(num_par, 1)
            r2 = np.random.rand(num_par, 1)
            velocities = w * velocities + c1 * r1 * (pBests - swarms) + c2 * r2 * (lBest_matrix - swarms)
            swarms += velocities

    total_time = time.time() - pso_start_time
    return gBest_feasible_bn, total_time

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == '__main__':
    # --- 実験設定 ---
    target_node_count = 1000     
    num_trials = 50               
    core_counts = [1, 2, 3, 4]
    methods = [
        {'name': 'Global', 'topo': 'global', 'restart': False},
        {'name': 'Spatial', 'topo': 'spatial', 'restart': False},
        {'name': 'Restart', 'topo': 'spatial', 'restart': True}
    ]
    # ----------------
    
    pso_params = {
        'num_par': 100, 'num_gen': 100, 
        'w_config': (0.9, 0.18), 'c1_config': (3.32, 0.44), 'c2_config': (0.88,3.66),
        'Pd_config': (6.63, 56.4), 
        'Pl_config': (73.9, 56447.8),  
        'Pr_config': (73.9, 56447.8)
    }
    constraints = {'delay_multiplier': 3.0, 'loss_constraint': 0.1, 'reliability_constraint': 0.95}

    result_path = savef.create_dir(dir_name="scalability_benchmark_3methods")
    csv_file = f"{result_path}/scalability_3methods_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    print(f"=== 並列化スケーラビリティ計測 (Node={target_node_count}) ===")
    print(f"Methods: {[m['name'] for m in methods]}")
    print(f"Cores: {core_counts}")

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['NodeCount', 'Trial', 'CoreCount', 'Method', 'BN', 'Time'])
        
        # 1. グラフ生成
        print("Generating Graph...", end="\r")
        Graph = create_graph.random_graph(num_node=target_node_count)
        if not nx.is_connected(Graph):
            largest_cc = max(nx.connected_components(Graph), key=len)
            nodes = list(largest_cc)
        else:
            nodes = list(Graph.nodes())
        
        problem_sets = []
        for _ in range(num_trials):
            s, d = random.sample(nodes, 2)
            problem_sets.append((s, d))
        print("Graph Generation Completed.")
        
        # 2. 実験ループ
        for t, (src, dst) in enumerate(problem_sets):
            print(f"\n--- Trial {t+1}/{num_trials} ---")
            
            for cores in core_counts:
                # 実際の物理コアチェック
                if cores > multiprocessing.cpu_count():
                    print(f"Warning: {cores} cores requested > available.")

                for method in methods:
                    m_name = method['name']
                    m_topo = method['topo']
                    m_restart = method['restart']
                    
                    print(f"  [{m_name}] Cores:{cores} ...", end="\r")
                    
                    bn, exec_time = run_pso_parallel_scalability(
                        Graph, src, dst, constraints, pso_params, 
                        num_cores=cores, 
                        topology=m_topo, 
                        enable_restart=m_restart
                    )
                    
                    print(f"  [{m_name}] Cores:{cores} | Time:{exec_time:.2f}s | BN:{bn:.2f}")
                    
                    writer.writerow([target_node_count, t+1, cores, m_name, bn, exec_time])
                    f.flush()

    print(f"\n全実験終了。結果は {csv_file} に保存されました。")