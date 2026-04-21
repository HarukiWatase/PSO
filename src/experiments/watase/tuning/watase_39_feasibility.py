# watase_40_feasibility_pareto.py
# 目的: 遅延制約倍率を変化させ、成功率・ボトルネック帯域・パレート解数を記録する

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
    particle, P_d, P_l, P_r = args
    path, is_valid = PathEncode(particle, worker_graph, worker_src, worker_dst)
    if not is_valid: return -1.0, False, -1.0
    bn, d, l, r = calculate_path_attributes_4d(worker_graph, path)
    
    max_delay = worker_constraints['delay_multiplier'] # 絶対値
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

# --- 厳密解法 (パレート数カウント機能付き) ---
def find_optimal_path_and_pareto_count(G, source, target, max_delay, max_loss_rate, min_reliability):
    # 初期化
    labels = {node: [] for node in G.nodes()}
    pq = []
    initial_label = (0.0, float('inf'), 0.0, 0.0) # (delay, neg_bn, loss, rel_cost)
    labels[source].append(initial_label)
    heappush(pq, (-initial_label[1], initial_label[0], initial_label[2], initial_label[3], source, initial_label))
    
    min_reliability_cost = -math.log(min_reliability) if min_reliability > 0 else float('inf')
    max_loss_log_cost = -math.log(1 - max_loss_rate) if max_loss_rate < 1 else float('inf')

    while pq:
        neg_bottle, d_curr, l_curr, r_curr, u, label_curr = heappop(pq)
        
        if label_curr not in labels[u]: continue
        
        for v in G.neighbors(u):
            edge = G.get_edge_data(u, v)
            d_new = d_curr + edge.get("delay", 0)
            b_new = min(-neg_bottle, edge.get("weight", 1))
            l_new = l_curr + edge.get("loss_log_cost", 0)
            r_new = r_curr + edge.get("reliability_cost", 0)
            
            if d_new > max_delay or l_new > max_loss_log_cost or r_new > min_reliability_cost: continue
            
            label_new = (d_new, b_new, l_new, r_new)
            
            # ドミナンスチェック (Pareto最適性の維持)
            is_dominated = False
            for d, b, l, r in labels[v]:
                # 遅延・損失・コストが既存より大きく、かつ帯域が小さいなら支配されている
                if d <= d_new and b >= b_new and l <= l_new and r <= r_new:
                    is_dominated = True; break
            if is_dominated: continue
            
            # 既存のラベルのうち、新ラベルに支配されるものを削除
            labels[v] = [lbl for lbl in labels[v] if not (lbl[0] >= d_new and lbl[1] <= b_new and lbl[2] >= l_new and lbl[3] >= r_new)]
            
            labels[v].append(label_new)
            heappush(pq, (-b_new, d_new, l_new, r_new, v, label_new))

    final_labels = labels.get(target, [])
    
    # ★パレート解の数 (到達可能な非劣解の数)
    num_pareto = len(final_labels)
    
    # 最良のボトルネック帯域を探す
    best_bottle = -1
    for d, b, l, r in final_labels:
        if b > best_bottle: best_bottle = b
        
    return best_bottle, num_pareto

# ==========================================
# 3. 並列PSO (戻り値拡張版)
# ==========================================
def run_pso_feasibility(Graph, src, dst, constraints, pso_params, num_cores, topology='spatial', enable_restart=True):
    try:
        min_delay_path = nx.dijkstra_path(Graph, source=src, target=dst, weight='delay')
        _, min_delay, _, _ = calculate_path_attributes_4d(Graph, min_delay_path)
        abs_max_delay = min_delay * constraints['delay_multiplier']
        constraints_for_worker = constraints.copy()
        constraints_for_worker['delay_multiplier'] = abs_max_delay
    except:
        return False, -1.0 # 失敗

    num_nodes = len(Graph.nodes()); num_par = pso_params['num_par']; num_gen = pso_params['num_gen']
    w_start, w_end = pso_params['w_config']
    c1_start, c1_end = pso_params['c1_config']; c2_start, c2_end = pso_params['c2_config']
    Pd_start, Pd_end = pso_params['Pd_config']; Pl_start, Pl_end = pso_params['Pl_config']; Pr_start, Pr_end = pso_params['Pr_config']
    
    RESTART_THRESHOLD = 20; restart_counter = 0

    swarms = np.random.uniform(1, 20, (num_par, num_nodes))
    velocities = np.zeros_like(swarms)
    pBests = np.copy(swarms); pBests_fitness = np.full(num_par, -float('inf'))
    gBest = swarms[0]; gBest_fitness = -float('inf')
    gBest_feasible_bn = -1; last_best_bn = -1.0
    
    found_feasible = False

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
            P_l = Pl_start + (Pl_end - Pl_start) * progress; P_r = Pr_start + (Pr_end - Pr_start) * progress
            
            args_list = [(swarms[j], P_d, P_l, P_r) for j in range(num_par)]
            results = pool.map(fitness_worker, args_list)
            
            current_fitness = np.zeros(num_par)
            for j, (fit, is_feas, bn) in enumerate(results):
                current_fitness[j] = fit
                if is_feas:
                    found_feasible = True
                    if bn > gBest_feasible_bn: gBest_feasible_bn = bn

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

    # ★戻り値追加: (成功可否, 最良BN)
    return found_feasible, gBest_feasible_bn

# ==========================================
# 4. メイン実行部
# ==========================================
if __name__ == '__main__':
    # --- 実験設定 ---
    target_node = 1000
    num_trials = 100   # 成功率算出用（時間はかかります）
    num_cores = 8
    
    methods = [
        {'name': 'Global', 'topo': 'global', 'restart': False},
        {'name': 'Spatial', 'topo': 'spatial', 'restart': False},
        {'name': 'Restart', 'topo': 'spatial', 'restart': True}
    ]
    
    # 厳しさのレベル（遅延倍率）
    delay_multipliers = [1.1, 1.2, 1.5, 2.0,2.5,3.0]
    
    pso_params = {
        'num_par': 100, 'num_gen': 100, 
        'w_config': (0.9, 0.18), 'c1_config': (3.32, 0.44), 'c2_config': (0.88,3.66),
        'Pd_config': (6.63, 56.4), 'Pl_config': (73.9, 56447.8), 'Pr_config': (73.9, 56447.8)
    }
    
    base_constraints = {'loss_constraint': 0.1, 'reliability_constraint': 0.95}

    result_path = savef.create_dir(dir_name="feasibility_benchmark_pareto")
    csv_file = f"{result_path}/feasibility_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    print(f"=== 成功率＆パレート数分析 (Node={target_node}, Trials={num_trials}) ===")

    # CSVヘッダ (Long Format)
    with open(csv_file, 'w', newline='') as f:
        csv.writer(f).writerow([
            'Multiplier', 'Trial', 
            'Pareto_Count', 'Exact_BN', 'Exact_Time', 
            'Method', 'Is_Success', 'Found_BN'
        ])

    for mult in delay_multipliers:
        print(f"\n>>> Delay Multiplier: {mult} <<<")
        
        for t in range(num_trials):
            # グラフ生成
            Graph = create_graph.random_graph(num_node=target_node)
            if not nx.is_connected(Graph):
                largest_cc = max(nx.connected_components(Graph), key=len)
                nodes = list(largest_cc)
            else: nodes = list(Graph.nodes())
            if len(nodes) < 2: continue
            src, dst = random.sample(nodes, 2)
            
            # 制約設定
            current_constraints = base_constraints.copy()
            current_constraints['delay_multiplier'] = mult

            print(f"  Trial {t+1}/{num_trials} [Exact]...", end="\r")
            
            # 1. 厳密解法 (パレート数取得のため)
            try:
                # 厳密解法用の絶対制約値を計算
                min_d_path = nx.dijkstra_path(Graph, source=src, target=dst, weight='delay')
                _, min_d, _, _ = calculate_path_attributes_4d(Graph, min_d_path)
                abs_max_delay = min_d * mult
                
                st = time.time()
                exact_bn, pareto_count = find_optimal_path_and_pareto_count(
                    Graph, src, dst, abs_max_delay, 
                    current_constraints['loss_constraint'], 
                    current_constraints['reliability_constraint']
                )
                exact_time = time.time() - st
            except:
                exact_bn = -1; pareto_count = 0; exact_time = 0
            
            # 2. 各PSO手法
            for m in methods:
                print(f"  Trial {t+1}/{num_trials} [{m['name']}]...", end="\r")
                
                is_success, found_bn = run_pso_feasibility(
                    Graph, src, dst, current_constraints, pso_params, num_cores,
                    topology=m['topo'], enable_restart=m['restart']
                )
                
                # 結果を1行ずつ保存 (Long Format)
                with open(csv_file, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        mult, t+1, 
                        pareto_count, exact_bn, exact_time, 
                        m['name'], is_success, found_bn
                    ])
                    
        print(f"  --> Multiplier {mult} Finished.")
                
    print(f"\n全実験終了。結果は {csv_file} に保存されました。")