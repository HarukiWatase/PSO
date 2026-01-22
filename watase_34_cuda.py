# watase_34_cuda.py
# 目的: MCOP/MBL問題に対し、CUDA（CuPy）を用いてPSOを高速化し、厳密解法と比較する

import networkx as nx
import numpy as np
try:
    import cupy as cp
    CUDA_AVAILABLE = True
    print("CuPy detected. GPU acceleration enabled.")
except ImportError:
    print("Warning: CuPy not found. Falling back to NumPy (CPU).")
    CUDA_AVAILABLE = False
    cp = np  # Fallback to NumPy

import create_graph_4 as create_graph
import savef
from datetime import datetime
import csv
from heapq import heappop, heappush
import time
import math
import random
import multiprocessing

# ==========================================
# 1. CPU側並列化用ワーカー設定 (PathEncode用)
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
    PathEncodeと属性計算はCPU側で実行（グラフ操作が複雑なため）
    Args:
        args (tuple): (particle_vector, P_d, P_l, P_r)
    Returns:
        tuple: (fitness, is_feasible, bottleneck_bandwidth)
    """
    particle, P_d, P_l, P_r = args
    
    # グローバル変数を使用
    # PathEncodeはCPU側で実行（グラフ操作が複雑なためGPU化が困難）
    path, is_valid = PathEncode(particle, worker_graph, worker_src, worker_dst)
    
    if not is_valid:
        return -1.0, False, -1.0
    
    # 属性計算（CPU側）
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
        
        # 優先度比較
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

# --- GPU用近傍計算関数 ---

def get_spatial_lbest_gpu(swarms_gpu, pBests_gpu, pBests_fitness_gpu, k=5):
    """
    GPU上で空間的近傍を計算（CuPyで行列演算を高速化）
    """
    # 距離の二乗を計算（GPU上で実行）
    sq_norms = cp.sum(swarms_gpu**2, axis=1)
    # ブロードキャストを使用した効率的な距離計算
    dist_sq = sq_norms[:, cp.newaxis] + sq_norms[cp.newaxis, :] - 2 * cp.dot(swarms_gpu, swarms_gpu.T)
    dist_sq = cp.maximum(dist_sq, 0)  # 数値安定性のため
    
    # k個の最近傍を取得（GPU上で実行）
    nearest_indices = cp.argpartition(dist_sq, kth=k, axis=1)[:, :k]
    
    # lBestを計算（CPUに転送せずGPU上で実行）
    lBest_matrix = cp.zeros_like(swarms_gpu)
    num_particles = swarms_gpu.shape[0]
    
    # GPU上でベクトル化された処理
    for i in range(num_particles):
        neighbors = nearest_indices[i].get()  # CPUに転送（小さなインデックス配列）
        best_local_idx = cp.argmax(pBests_fitness_gpu[neighbors]).get()
        lBest_matrix[i] = pBests_gpu[neighbors[best_local_idx]]
    
    return lBest_matrix

def get_global_lbest_gpu(swarms_gpu, gBest_gpu):
    """
    GPU上でグローバルlBestを計算
    """
    return cp.tile(gBest_gpu, (swarms_gpu.shape[0], 1))

# --- 厳密解法（変更なし） ---
def find_optimal_path_by_label_correcting_4d(G, source, target, max_delay, max_loss_rate, min_reliability):
    # 厳密解法は並列化が難しいため、シングルスレッド実行
    labels = {node: [] for node in G.nodes()}
    pq = []
    initial_label = (0.0, float('inf'), 0.0, 0.0) # (delay, neg_bn, loss, rel_cost)
    labels[source].append(initial_label)
    
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
# 3. CUDA並列化 PSO シミュレーション関数
# ==========================================

def run_pso_cuda(Graph, src, dst, constraints, pso_params, topology='spatial', enable_restart=True):
    """
    CUDA（CuPy）を用いた並列PSO
    - 粒子の評価（PathEncode + 属性計算）: CPU側で並列化
    - PSO更新処理（位置更新、速度更新、近傍計算）: GPU側で並列化
    """
    # 事前経路計算 (遅延制約の上限設定用)
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

    # ★粒子初期化（GPUメモリ上に配置）
    # CPU側で生成してからGPUに転送
    swarms_cpu = np.random.uniform(1, 20, (num_par, num_nodes))
    swarms_gpu = cp.asarray(swarms_cpu)  # GPUメモリに転送
    velocities_gpu = cp.zeros_like(swarms_gpu)
    pBests_gpu = cp.array(swarms_gpu)  # GPU上でコピー
    pBests_fitness_cpu = np.full(num_par, -float('inf'))
    pBests_fitness_gpu = cp.asarray(pBests_fitness_cpu)
    
    gBest_cpu = swarms_cpu[0]
    gBest_gpu = cp.asarray(gBest_cpu)
    gBest_fitness = -float('inf')
    gBest_feasible_bn = -1
    last_best_bn = -1.0 
    
    # CPU側並列評価用のPool作成（PathEncode用）
    num_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(
        processes=min(num_cores, num_par),
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
                    swarms_cpu = np.random.uniform(1, 20, (num_par, num_nodes))
                    swarms_gpu = cp.asarray(swarms_cpu)
                    velocities_gpu = cp.zeros_like(swarms_gpu)
                    if gBest_gpu is not None: 
                        swarms_gpu[0] = gBest_gpu  # Keep Elite
                    pBests_gpu = cp.array(swarms_gpu)
                    pBests_fitness_cpu = np.full(num_par, -float('inf'))
                    pBests_fitness_gpu = cp.asarray(pBests_fitness_cpu)
                    restart_counter = 0

            # --- Update Parameters ---
            progress = i / num_gen
            w = w_start - (w_start - w_end) * progress
            c1 = c1_start - (c1_start - c1_end) * progress
            c2 = c2_start + (c2_end - c2_start) * progress
            P_d = Pd_start + (Pd_end - Pd_start) * progress
            P_l = Pl_start + (Pl_end - Pl_start) * progress
            P_r = Pr_start + (Pr_end - Pr_start) * progress
            
            # --- ★CPU側並列評価 (Evaluation) ---
            # GPU上の粒子をCPUに転送して評価
            swarms_cpu = cp.asnumpy(swarms_gpu)  # GPU -> CPU
            
            args_list = [
                (swarms_cpu[j], P_d, P_l, P_r) for j in range(num_par)
            ]
            
            # CPU側で並列実行（PathEncodeはグラフ操作が複雑なためCPU側）
            results = pool.map(fitness_worker, args_list)
            
            # 結果を集計
            current_fitness_cpu = np.zeros(num_par)
            for j, (fit, is_feas, bn) in enumerate(results):
                current_fitness_cpu[j] = fit
                
                # gBest (Feasible) の更新
                if is_feas and bn > gBest_feasible_bn:
                    gBest_feasible_bn = bn

            # CPU上のfitnessをGPUに転送
            current_fitness_gpu = cp.asarray(current_fitness_cpu)

            # --- ★GPU上でpBest / gBest (Fitness based) 更新 ---
            # GPU上でブールインデックスを使用した高速更新
            update_mask = current_fitness_gpu > pBests_fitness_gpu
            pBests_gpu[update_mask] = swarms_gpu[update_mask]
            pBests_fitness_gpu[update_mask] = current_fitness_gpu[update_mask]
            
            # gBest更新（GPU上でargmaxを計算）
            current_best_idx = int(cp.argmax(current_fitness_gpu).get())
            if current_fitness_cpu[current_best_idx] > gBest_fitness:
                gBest_fitness = current_fitness_cpu[current_best_idx]
                gBest_gpu = swarms_gpu[current_best_idx].copy()
                gBest_cpu = swarms_cpu[current_best_idx]
            
            # --- ★GPU上でTopology Selection ---
            if topology == 'spatial':
                lBest_matrix_gpu = get_spatial_lbest_gpu(swarms_gpu, pBests_gpu, pBests_fitness_gpu, k=5)
            else:  # global
                lBest_matrix_gpu = get_global_lbest_gpu(swarms_gpu, gBest_gpu)

            # --- ★GPU上でVelocity Update ---
            # 乱数生成もGPU上で実行（高速化）
            r1_gpu = cp.random.rand(num_par, 1)
            r2_gpu = cp.random.rand(num_par, 1)
            
            # GPU上で速度更新を計算
            velocities_gpu = (w * velocities_gpu + 
                             c1 * r1_gpu * (pBests_gpu - swarms_gpu) + 
                             c2 * r2_gpu * (lBest_matrix_gpu - swarms_gpu))
            
            # GPU上で位置更新
            swarms_gpu += velocities_gpu

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
    
    result_path = savef.create_dir(dir_name="comparison_benchmark_cuda")
    csv_file = f"{result_path}/compare_cuda_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    if CUDA_AVAILABLE:
        print(f"Using GPU: {cp.cuda.Device().compute_capability}")
    else:
        print("Warning: Running on CPU (CuPy not available)")
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'NodeCount', 'Trial', 
            'Exact_BN', 'Exact_Time', 
            'Global_BN', 'Global_Time', 
            'Spatial_BN', 'Spatial_Time', 
            'Restart_BN', 'Restart_Time'
        ])
        
        print("=== 4手法 比較ベンチマーク（CUDA並列化版）開始 ===")
        
        for N in node_counts:
            print(f"\n--- Node Count: {N} ---")
            
            for t in range(num_trials):
                # 1. グラフ生成
                Graph = create_graph.random_graph(num_node=N)
                if not nx.is_connected(Graph):
                    largest_cc = max(nx.connected_components(Graph), key=len)
                    nodes = list(largest_cc)
                else:
                    nodes = list(Graph.nodes())
                if len(nodes) < 2: continue
                src, dst = random.sample(nodes, 2)
                
                # --- Method 1: Exact (Baseline) - 厳密解法はそのまま ---
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
                
                # --- Method 2: Global PSO (CUDA) ---
                print(f"Trial {t+1} - Global PSO (CUDA)...", end="\r")
                glob_bn, glob_time = run_pso_cuda(
                    Graph, src, dst, constraints, pso_params, 
                    topology='global', enable_restart=False
                )
                
                # --- Method 3: Spatial PSO (CUDA) ---
                print(f"Trial {t+1} - Spatial PSO (CUDA)...", end="\r")
                spat_bn, spat_time = run_pso_cuda(
                    Graph, src, dst, constraints, pso_params, 
                    topology='spatial', enable_restart=False
                )

                # --- Method 4: Spatial + Restart (CUDA) ---
                print(f"Trial {t+1} - Restart PSO (CUDA)...", end="\r")
                rest_bn, rest_time = run_pso_cuda(
                    Graph, src, dst, constraints, pso_params, 
                    topology='spatial', enable_restart=True
                )
                
                # Log
                print(f"Trial {t+1}: Exact={exact_bn:.2f}({exact_time:.2f}s) | "
                      f"Glob={glob_bn:.2f}({glob_time:.2f}s) | "
                      f"Spat={spat_bn:.2f}({spat_time:.2f}s) | "
                      f"Rest={rest_bn:.2f}({rest_time:.2f}s)")
                
                # CSV Save
                writer.writerow([
                    N, t+1, 
                    exact_bn, exact_time, 
                    glob_bn, glob_time, 
                    spat_bn, spat_time,
                    rest_bn, rest_time
                ])
                # バッファフラッシュ（途中経過を確実に保存）
                f.flush()

    print(f"\n全実験終了。結果は {csv_file} に保存されました。")
