# watase_37_compare_all.py
# 目的: 4つの手法（厳密解法, 従来PSO, 空間近傍PSO, 提案手法）を同一条件で比較する

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
import cProfile
import pstats
# ==========================================
# 1. 共通関数群
# ==========================================

def PathEncode(Particle, Graph, src, dst):
    path = [src]
    current_node = src
    visited = {src}
    limit_len = len(Particle)
    while current_node != dst:
        # List comprehension for speed
        neighbors = [n for n in Graph.neighbors(current_node) if n not in visited]
        if not neighbors: return path, False
        best_neighbor = -1; highest_prio = -1.0
        for neighbor in neighbors:
            if neighbor < limit_len:
                prio = Particle[neighbor]
                if prio > highest_prio: highest_prio = prio; best_neighbor = neighbor
        if best_neighbor == -1: return path, False
        current_node = best_neighbor; path.append(current_node); visited.add(current_node)
    return path, True

def calculate_path_attributes_4d(G, path):
    if not path or len(path) < 2: return 0, float('inf'), 1.0, 0.0
    bottleneck = float('inf'); total_delay = 0; total_loss_log_cost = 0; total_reliability_cost = 0
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

# 空間的近傍 (lBest = 近い上位k個のベスト)
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

# 従来型 (Global Topology: lBest = gBest)
def get_global_lbest(swarms, gBest):
    # 全員がgBestを目指す
    return np.tile(gBest, (len(swarms), 1))

# --- 厳密解法 ---
def find_optimal_path_by_label_correcting_4d(G, source, target, max_delay, max_loss_rate, min_reliability):
    labels = {node: [] for node in G.nodes()}; pq = []
    initial_label = (0.0, float('inf'), 0.0, 0.0)
    labels[source].append(initial_label)
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
            if d_new > max_delay or l_new > max_loss_log_cost or r_new > min_reliability_cost: continue
            label_new = (d_new, b_new, l_new, r_new)
            is_dominated = False
            for d, b, l, r in labels[v]:
                if d <= d_new and b >= b_new and l <= l_new and r <= r_new: is_dominated = True; break
            if is_dominated: continue
            labels[v] = [lbl for lbl in labels[v] if not (lbl[0] >= d_new and lbl[1] <= b_new and lbl[2] >= l_new and lbl[3] >= r_new)]
            labels[v].append(label_new)
            heappush(pq, (-b_new, d_new, l_new, r_new, v, label_new))

    final_labels = labels.get(target, [])
    if not final_labels: return -1
    best_bottle = -1
    for d, b, l, r in final_labels:
        if b > best_bottle: best_bottle = b
    return best_bottle

# ==========================================
# 2. 統合PSOシミュレーション関数
# ==========================================
def run_pso_variant(Graph, src, dst, constraints, pso_params, topology='spatial', enable_restart=True):
    """
    Args:
        topology (str): 'global' or 'spatial'
        enable_restart (bool): Trueなら停滞検知＆爆発を行う
    """
    max_delay = constraints['delay_multiplier']
    try:
        min_delay_path = nx.dijkstra_path(Graph, source=src, target=dst, weight='delay')
        _, min_delay, _, _ = calculate_path_attributes_4d(Graph, min_delay_path)
        max_delay = min_delay * constraints['delay_multiplier']
    except: return -1.0, 0.0

    max_loss = constraints['loss_constraint']
    min_rel = constraints['reliability_constraint']

    num_nodes = len(Graph.nodes())
    num_par = pso_params['num_par']
    num_gen = pso_params['num_gen']
    
    # パラメータ展開
    w_start, w_end = pso_params['w_config']; c1_start, c1_end = pso_params['c1_config']
    c2_start, c2_end = pso_params['c2_config']; Pd_start, Pd_end = pso_params['Pd_config']
    Pl_start, Pl_end = pso_params['Pl_config']; Pr_start, Pr_end = pso_params['Pr_config']
    
    TIME_LIMIT_SEC = pso_params['time_limit_sec']
    pso_start_time = time.time()
    
    # Restart関連
    RESTART_THRESHOLD = 20
    restart_counter = 0

    swarms = np.random.uniform(1, 20, (num_par, num_nodes))
    velocities = np.zeros_like(swarms)
    pBests = np.copy(swarms); pBests_fitness = np.full(num_par, -float('inf'))
    gBest = swarms[0]; gBest_fitness = -float('inf')
    gBest_feasible_bn = -1
    last_best_bn = -1.0 

    for i in range(num_gen):
        if time.time() - pso_start_time > TIME_LIMIT_SEC: break

        # --- 1. Restart Logic (条件付き実行) ---
        if enable_restart:
            if gBest_feasible_bn > last_best_bn:
                restart_counter = 0; last_best_bn = gBest_feasible_bn
            else:
                restart_counter += 1
                
            if restart_counter >= RESTART_THRESHOLD:
                # Explosion
                swarms = np.random.uniform(1, 20, (num_par, num_nodes))
                velocities = np.zeros_like(swarms)
                if gBest is not None: swarms[0] = gBest # Keep Elite
                pBests = np.copy(swarms); pBests_fitness = np.full(num_par, -float('inf'))
                restart_counter = 0

        # --- 2. Update Parameters ---
        progress = i / num_gen
        w = w_start - (w_start - w_end) * progress
        c1 = c1_start - (c1_start - c1_end) * progress
        c2 = c2_start + (c2_end - c2_start) * progress
        P_d = Pd_start + (Pd_end - Pd_start) * progress
        P_l = Pl_start + (Pl_end - Pl_start) * progress
        P_r = Pr_start + (Pr_end - Pr_start) * progress
        
        current_fitness = np.zeros(num_par)
        
        # --- 3. Evaluate Swarm ---
        for j in range(num_par):
            path, is_valid = PathEncode(swarms[j], Graph, src, dst)
            if not is_valid: current_fitness[j] = -1.0; continue
            
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

        update_indices = current_fitness > pBests_fitness
        pBests[update_indices] = swarms[update_indices]; pBests_fitness[update_indices] = current_fitness[update_indices]
        current_best_idx = np.argmax(current_fitness)
        if current_fitness[current_best_idx] > gBest_fitness:
            gBest_fitness = current_fitness[current_best_idx]; gBest = swarms[current_best_idx]
        
        # --- 4. Topology Selection ---
        if topology == 'spatial':
            lBest_matrix = get_spatial_lbest(swarms, pBests, pBests_fitness, k=5)
        else: # global
            lBest_matrix = get_global_lbest(swarms, gBest)

        # --- 5. Velocity Update ---
        r1, r2 = np.random.rand(2, num_par, 1)
        velocities = w * velocities + c1 * r1 * (pBests - swarms) + c2 * r2 * (lBest_matrix - swarms)
        swarms += velocities

    total_time = time.time() - pso_start_time
    return gBest_feasible_bn, total_time

# ==========================================
# 3. Main Comparison Loop
# ==========================================
if __name__ == '__main__':
    # Tuned Parameters (N=300~1000 Optimized)
    pso_params = {
        'num_par': 100, 'num_gen': 100,
        'convergence_gen': 100,
        'time_limit_sec': 1200, 
        'w_config': (0.9, 0.18), 'c1_config': (3.32, 0.44), 'c2_config': (0.88,3.66),
        # 2. ペナルティ
        # Delay: 序盤は0.1(無視) -> 終盤は100.0(絶対遵守)
        'Pd_config': (6.63, 56.4), 
        'Pl_config': (73.9, 56447.8),  
        'Pr_config': (73.9, 56447.8)
    }
    
    constraints = {'delay_multiplier': 3.0, 'loss_constraint': 0.1, 'reliability_constraint': 0.95}
    
    # 実験設定
    node_counts = [1000] # ノード数を変えて傾向を見る
    num_trials = 1                     # 各10回
    
    result_path = savef.create_dir(dir_name="comparison_benchmark")
    csv_file = f"{result_path}/compare_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(csv_file, 'w', newline='') as f:
        # Header: 全手法のBNとTimeを横並びに記録
        writer = csv.writer(f)
        writer.writerow([
            'NodeCount', 'Trial', 
            'Exact_BN', 'Exact_Time', 
            'Global_BN', 'Global_Time', 
            'Spatial_BN', 'Spatial_Time', 
            'Restart_BN', 'Restart_Time'
        ])
        
        print("=== 4手法 比較ベンチマーク開始 ===")
        
        for N in node_counts:
            print(f"\n--- Node Count: {N} ---")
            
            for t in range(num_trials):
                # 1. グラフ生成 (共通)
                Graph = create_graph.random_graph(num_node=N)
                if not nx.is_connected(Graph):
                    largest_cc = max(nx.connected_components(Graph), key=len)
                    nodes = list(largest_cc)
                else:
                    nodes = list(Graph.nodes())
                if len(nodes) < 2: continue
                src, dst = random.sample(nodes, 2)
                
                # --- Method 1: Exact (Baseline) ---
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
                

                # --- Method 2: Global PSO (Conventional) ---
                
                # **1. cProfile/pstats/osモジュールをインポート**
                import cProfile
                import pstats 
                import io # 出力バッファリング用
                import sys # 標準出力のリダイレクト用

                profiler = cProfile.Profile()
                
                # 実行関数の定義 (run_pso_variantはそのまま)
                def profile_target():
                    return run_pso_variant(
                        Graph, src, dst, constraints, pso_params, 
                        topology='global', enable_restart=False
                    )

                # **2. プロファイリングを実行**
                profiler.enable()
                # runcallではなく、直接関数を呼び出す方法に変更 (よりシンプルで確実)
                glob_bn, glob_time = profile_target()
                profiler.disable()

                # **3. 結果を文字列としてキャプチャし、ファイルに書き出す**
                profile_filename = f'{result_path}/profile_output_global_{N}_{t+1}.txt'
                
                stats = pstats.Stats(profiler)
                stats.strip_dirs()
                stats.sort_stats('tottime') 
                
                # io.StringIOを使用して、一時的にメモリ上に出力をキャプチャ
                s = io.StringIO()
                # sys.stdout を一時的に s にリダイレクトして print_stats の出力を捕まえる
                sys.stdout = s
                
                # print_stats を引数なしで実行 (これが最も古いバージョンまで互換性がある)
                stats.print_stats()
                
                # 標準出力 (sys.stdout) を元に戻す
                sys.stdout = sys.__stdout__ 
                
                # キャプチャした文字列をファイルに書き込む
                with open(profile_filename, 'w') as out_f:
                    out_f.write(s.getvalue())
                    
                print(f"Global PSOのプロファイリング結果を {profile_filename} に保存しました。")

                # --- Method 3: Spatial PSO (Step 1) ---
                # topology='spatial', restart=False
                spat_bn, spat_time = run_pso_variant(
                    Graph, src, dst, constraints, pso_params, 
                    topology='spatial', enable_restart=False
                )

                # --- Method 4: Spatial + Restart (Proposed) ---
                # topology='spatial', restart=True
                rest_bn, rest_time = run_pso_variant(
                    Graph, src, dst, constraints, pso_params, 
                    topology='spatial', enable_restart=True
                )
                
                # Log
                print(f"Trial {t+1}: Exact={exact_bn} | Glob={glob_bn} | Spat={spat_bn} | Rest={rest_bn}")
                
                # CSV Save
                writer.writerow([
                    N, t+1, 
                    exact_bn, exact_time, 
                    glob_bn, glob_time, 
                    spat_bn, spat_time, 
                    rest_bn, rest_time
                ])

    print(f"\n全実験終了。結果は {csv_file} に保存されました。")