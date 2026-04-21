# watase_34.py
# 目的: MCOP/MBL問題に対し、Multiprocessingを用いてPSOを高速化し、厳密解法と比較する

import networkx as nx
import numpy as np
import create_graph_4 as create_graph
import savef
from datetime import datetime
import csv
from heapq import heappop, heappush
import time
import math
import random
import multiprocessing # 並列化用

# ==========================================
# 1. 並列化用ワーカー設定 (Global Scope)
# ==========================================

# ワーカープロセス内でデータを保持するためのグローバル変数
worker_graph = None
worker_src = None
worker_dst = None
worker_constraints = None

def init_worker(G, s, d, const):
    """
    ワーカープロセスの初期化関数。
    巨大なGraphオブジェクトを毎回Pickle化して送るオーバーヘッドを防ぐため、
    プロセス起動時に一度だけグローバル変数としてセットする。
    """
    global worker_graph, worker_src, worker_dst, worker_constraints
    worker_graph = G
    worker_src = s
    worker_dst = d
    worker_constraints = const

def fitness_worker(args):
    """
    並列実行される評価関数（粒子1つ分を受け取って計算）
    Args:
        args (tuple): (particle_vector, P_d, P_l, P_r)
    Returns:
        tuple: (fitness, is_feasible, bottleneck_bandwidth)
    """
    particle, P_d, P_l, P_r = args
    
    # グローバル変数を使用
    # ここがボトルネックだった PathEncode
    path, is_valid = PathEncode(particle, worker_graph, worker_src, worker_dst)
    
    if not is_valid:
        return -1.0, False, -1.0
    
    # 属性計算
    bn, d, l, r = calculate_path_attributes_4d(worker_graph, path)
    
    # 制約値の取得
    max_delay = worker_constraints['delay_multiplier']
    max_loss = worker_constraints['loss_constraint']
    min_rel = worker_constraints['reliability_constraint']
    
    # ペナルティ計算
    penalty = 0
    if d > max_delay: penalty += P_d * (d - max_delay)
    if l > max_loss: penalty += P_l * (l - max_loss)
    if r < min_rel: penalty += P_r * (min_rel - r)
    
    fitness = bn - penalty
    is_feasible = (d <= max_delay and l <= max_loss and r >= min_rel)
    
    return fitness, is_feasible, bn

# ==========================================
# 2. 共通関数群 (ロジック)
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
        
        # 優先度比較 (ここがPythonループのボトルネック箇所だが、並列化で分散させる)
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

# --- 近傍計算関数 ---

def get_spatial_lbest(swarms, pBests, pBests_fitness, k=5):
    # 行列演算なので高速 (メインプロセスで実行)
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

# --- 厳密解法 ---
def find_optimal_path_by_label_correcting_4d(G, source, target, max_delay, max_loss_rate, min_reliability):
    # 厳密解法は並列化が難しいため、シングルスレッド実行
    labels = {node: [] for node in G.nodes()}
    pq = []
    initial_label = (0.0, float('inf'), 0.0, 0.0) # (delay, neg_bn, loss, rel_cost)
    labels[source].append(initial_label)
    
    # Heap format: (-neg_bn, delay, loss, rel_cost, node, label_tuple)
    # Note: Using neg_bn allows us to use min-heap to pop max bottleneck if we adjust logic, 
    # but here we follow standard cost minimization. Wait, Dijkstra minimizes.
    # We want max Bandwidth. So we minimize negative bandwidth.
    # Here logic: standard label correcting checks dominance.
    
    heappush(pq, (-initial_label[1], initial_label[0], initial_label[2], initial_label[3], source, initial_label))
    
    min_reliability_cost = -math.log(min_reliability) if min_reliability > 0 else float('inf')
    max_loss_log_cost = -math.log(1 - max_loss_rate) if max_loss_rate < 1 else float('inf')

    best_bottle = -1

    while pq:
        neg_bottle, d_curr, l_curr, r_curr, u, label_curr = heappop(pq)
        
        # Label cleaning check
        if label_curr not in labels[u]: continue
        
        for v in G.neighbors(u):
            edge = G.get_edge_data(u, v)
            d_new = d_curr + edge.get("delay", 0)
            b_new = min(-neg_bottle, edge.get("weight", 1))
            l_new = l_curr + edge.get("loss_log_cost", 0)
            r_new = r_curr + edge.get("reliability_cost", 0)
            
            if d_new > max_delay or l_new > max_loss_log_cost or r_new > min_reliability_cost: continue
            
            label_new = (d_new, b_new, l_new, r_new)
            
            # Dominance check
            is_dominated = False
            for d, b, l, r in labels[v]:
                # Dominance: new label is worse if Delay>=, BN<=, Loss>=, RelCost>=
                if d <= d_new and b >= b_new and l <= l_new and r <= r_new:
                    is_dominated = True
                    break
            if is_dominated: continue
            
            # Remove dominated labels
            labels[v] = [lbl for lbl in labels[v] if not (lbl[0] >= d_new and lbl[1] <= b_new and lbl[2] >= l_new and lbl[3] >= r_new)]
            labels[v].append(label_new)
            
            # Sort order priority: Maximize BN -> Minimize NegBN
            heappush(pq, (-b_new, d_new, l_new, r_new, v, label_new))

    final_labels = labels.get(target, [])
    if not final_labels: return -1
    
    for d, b, l, r in final_labels:
        if b > best_bottle: best_bottle = b
    return best_bottle


# ==========================================
# 3. 並列化 PSO シミュレーション関数
# ==========================================

def run_pso_parallel(Graph, src, dst, constraints, pso_params, topology='spatial', enable_restart=True):
    """
    Multiprocessingを用いた並列PSO
    """
    # 事前経路計算 (遅延制約の上限設定用)
    try:
        min_delay_path = nx.dijkstra_path(Graph, source=src, target=dst, weight='delay')
        _, min_delay, _, _ = calculate_path_attributes_4d(Graph, min_delay_path)
        abs_max_delay = min_delay * constraints['delay_multiplier']
        # Workerに渡す制約辞書を更新（倍率ではなく絶対値にする）
        constraints_for_worker = constraints.copy()
        constraints_for_worker['delay_multiplier'] = abs_max_delay
    except:
        return -1.0, 0.0

    num_nodes = len(Graph.nodes())
    num_par = pso_params['num_par']
    num_gen = pso_params['num_gen']
    
    # パラメータ展開
    w_start, w_end = pso_params['w_config']
    c1_start, c1_end = pso_params['c1_config']
    c2_start, c2_end = pso_params['c2_config']
    Pd_start, Pd_end = pso_params['Pd_config']
    Pl_start, Pl_end = pso_params['Pl_config']
    Pr_start, Pr_end = pso_params['Pr_config']
    
    TIME_LIMIT_SEC = pso_params['time_limit_sec']
    pso_start_time = time.time()
    
    # Restart関連
    RESTART_THRESHOLD = 20
    restart_counter = 0

    # 粒子初期化
    swarms = np.random.uniform(1, 20, (num_par, num_nodes))
    velocities = np.zeros_like(swarms)
    pBests = np.copy(swarms)
    pBests_fitness = np.full(num_par, -float('inf'))
    gBest = swarms[0]
    gBest_fitness = -float('inf')
    gBest_feasible_bn = -1
    last_best_bn = -1.0 

    # ★並列化: コア数取得 (MacBook Air M2 = 8 cores)
    # 安全のため物理コア数を使うか、cpu_countそのままを使う
    num_cores = 4

    # ★ Poolの作成 (コンテキストマネージャ)
    # initializerを使ってGraph等を共有メモリ(Copy-on-Write)的に渡す
    with multiprocessing.Pool(
        processes=num_cores,
        initializer=init_worker,
        initargs=(Graph, src, dst, constraints_for_worker)
    ) as pool:

        for i in range(num_gen):
            if time.time() - pso_start_time > TIME_LIMIT_SEC: break

            # --- Restart Logic ---
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

            # --- Update Parameters ---
            progress = i / num_gen
            w = w_start - (w_start - w_end) * progress
            c1 = c1_start - (c1_start - c1_end) * progress
            c2 = c2_start + (c2_end - c2_start) * progress
            P_d = Pd_start + (Pd_end - Pd_start) * progress
            P_l = Pl_start + (Pl_end - Pl_start) * progress
            P_r = Pr_start + (Pr_end - Pr_start) * progress
            
            # --- ★並列評価 (Evaluation) ---
            # 粒子ごとに現在のペナルティ係数と共に引数リストを作成
            args_list = [
                (swarms[j], P_d, P_l, P_r) for j in range(num_par)
            ]
            
            # 並列実行 (PathEncodeがここで8並列で走る)
            results = pool.map(fitness_worker, args_list)
            
            # 結果を集計
            current_fitness = np.zeros(num_par)
            for j, (fit, is_feas, bn) in enumerate(results):
                current_fitness[j] = fit
                
                # gBest (Feasible) の更新
                if is_feas and bn > gBest_feasible_bn:
                    gBest_feasible_bn = bn

            # pBest / gBest (Fitness based) 更新
            update_indices = current_fitness > pBests_fitness
            pBests[update_indices] = swarms[update_indices]
            pBests_fitness[update_indices] = current_fitness[update_indices]
            
            current_best_idx = np.argmax(current_fitness)
            if current_fitness[current_best_idx] > gBest_fitness:
                gBest_fitness = current_fitness[current_best_idx]
                gBest = swarms[current_best_idx]
            
            # --- Topology Selection ---
            if topology == 'spatial':
                lBest_matrix = get_spatial_lbest(swarms, pBests, pBests_fitness, k=5)
            else: # global
                lBest_matrix = get_global_lbest(swarms, gBest)

            # --- Velocity Update ---
            r1 = np.random.rand(num_par, 1)
            r2 = np.random.rand(num_par, 1)
            velocities = w * velocities + c1 * r1 * (pBests - swarms) + c2 * r2 * (lBest_matrix - swarms)
            swarms += velocities

    total_time = time.time() - pso_start_time
    return gBest_feasible_bn, total_time


# ==========================================
# 4. Main Comparison Loop
# ==========================================
if __name__ == '__main__':
    # Tuned Parameters
    pso_params = {
        'num_par': 100, 'num_gen': 100,
        'convergence_gen': 100,
        'time_limit_sec': 9900, 
        'w_config': (0.9, 0.18), 'c1_config': (3.32, 0.44), 'c2_config': (0.88,3.66),
        'Pd_config': (6.63, 56.4), 
        'Pl_config': (73.9, 56447.8),  
        'Pr_config': (73.9, 56447.8)
    }
    
    constraints = {'delay_multiplier': 3.0, 'loss_constraint': 0.1, 'reliability_constraint': 0.95}
    
    # 実験設定
    node_counts = [1000] 
    num_trials = 10
    
    result_path = savef.create_dir(dir_name="comparison_benchmark_parallel")
    csv_file = f"{result_path}/compare_parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    print(f"Using {multiprocessing.cpu_count()} cores for PSO.")

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'NodeCount', 'Trial', 
            'Exact_BN', 'Exact_Time', 
            'Global_BN', 'Global_Time', 
            'Spatial_BN', 'Spatial_Time', 
            'Restart_BN', 'Restart_Time'
        ])
        
        print("=== 4手法 比較ベンチマーク（並列化版）開始 ===")
        
        for N in node_counts:
            print(f"\n--- Node Count: {N} ---")
            
            for t in range(num_trials):
                # 1. グラフ生成
                Graph = create_graph.random_graph(num_node=N,prob=10/N)
                if not nx.is_connected(Graph):
                    largest_cc = max(nx.connected_components(Graph), key=len)
                    nodes = list(largest_cc)
                else:
                    nodes = list(Graph.nodes())
                if len(nodes) < 2: continue
                src, dst = random.sample(nodes, 2)
                
                # --- Method 1: Exact (Baseline) - 厳密解法は並列化できないのでそのまま ---
                print(f"Trial {t+1} - Exact...", end="\r")
                try:
                    d_path = nx.dijkstra_path(Graph, source=src, target=dst, weight='delay')
                    _, min_d, _, _ = calculate_path_attributes_4d(Graph, d_path)
                    abs_max_delay = min_d * constraints['delay_multiplier']
                    
                    st = time.time()
                    exact_bn = find_optimal_path_by_label_correcting_4d(
                        Graph, src, dst, abs_max_delay, 
                        constraints['loss_constraint'], constraints['reliability_constraint']
                    )
                    exact_time = time.time() - st
                except:
                    exact_bn = -1; exact_time = 0
                
                # --- Method 2: Global PSO (Parallel) ---
                print(f"Trial {t+1} - Global PSO (Parallel)...", end="\r")
                glob_bn, glob_time = run_pso_parallel(
                    Graph, src, dst, constraints, pso_params, 
                    topology='global', enable_restart=False
                )
                
                # --- Method 3: Spatial PSO (Parallel) ---
                print(f"Trial {t+1} - Spatial PSO (Parallel)...", end="\r")
                spat_bn, spat_time = run_pso_parallel(
                    Graph, src, dst, constraints, pso_params, 
                    topology='spatial', enable_restart=False
                )

                # --- Method 4: Spatial + Restart (Parallel) ---
                print(f"Trial {t+1} - Restart PSO (Parallel)...", end="\r")
                rest_bn, rest_time = run_pso_parallel(
                    Graph, src, dst, constraints, pso_params, 
                    topology='spatial', enable_restart=True
                )
                
                # Log
                print(f"Trial {t+1}: Exact={exact_bn:.2f}({exact_time:.2f}s) | "
                      f"Glob={glob_bn:.2f}({glob_time:.2f}s) | "
                      f"Rest={rest_bn:.2f}({rest_time:.2f}s)")
                
                # CSV Save
                writer.writerow([
                    N, t+1, 
                    exact_bn, exact_time, 
                    glob_bn, glob_time, 
                    rest_bn, rest_time
                ])
                # バッファフラッシュ（途中経過を確実に保存）
                f.flush()

    print(f"\n全実験終了。結果は {csv_file} に保存されました。")