"""Comparison experiment between immediate and delayed PSO references.

This file is migrated from `watase_25.py` and benchmarks three approaches:
label-correcting exact search, PSO with immediate pBest/gBest reference, and
PSO with delayed (one-generation-old) reference.
"""

# watase_25_comparison.py
# (ベース: watase_24.py)

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
[変更点 (watase_25)]
- 厳密解法、即時参照PSO、遅延参照PSO の3つを比較するため、
  PSOの実行ロジックを `_run_pso_loop` 関数に分離。
- `simulation` 関数が、この `_run_pso_loop` を
  `mode='immediate'` と `mode='delayed'` で2回呼び出すように変更。
- main関数も、3つのアルゴリズムの結果を
  progress_log.csv と summary.csv に書き出すよう変更。
"""

# (PathEncode, calculate_path_attributes_4d, find_optimal_path_by_label_correcting_4d は watase_24.py と同一)
def PathEncode(Particle, Graph, src, dst):
    path = [src]
    current_node = src
    visited = {src}
    particle_values = Particle # Assuming nodes are 0..N-1
    while current_node != dst:
        neighbors = [n for n in Graph.neighbors(current_node) if n not in visited]
        if not neighbors: return path, False
        highest_prio = -1; next_node = -1
        for neighbor in neighbors:
            if neighbor < len(particle_values): # Check index boundary
                neighbor_prio = particle_values[neighbor]
                if neighbor_prio > highest_prio:
                    highest_prio = neighbor_prio; next_node = neighbor
        if next_node == -1: return path, False
        current_node = next_node
        path.append(current_node); visited.add(current_node)
    return path, path[-1] == dst

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

def find_optimal_path_by_label_correcting_4d(G, source, target, max_delay, max_loss_rate, min_reliability):
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


# ★★★ 変更点: PSOの実行ループを独立した関数に分離 ★★★
def _run_pso_loop(Graph, src, dst, constraints, pso_params, max_delay, max_loss, min_rel, reference_mode='immediate'):
    """
    PSOのメインループを実行する内部関数。
    reference_mode: 'immediate' は同世代参照、'delayed' は1世代遅延参照。
    """
    
    pso_start_time = time.time()
    pso_progress_history = [] 

    num_nodes, num_par, num_gen = len(Graph.nodes()), pso_params['num_par'], pso_params['num_gen']
    w_start, w_end = pso_params['w_config']; c1_start, c1_end = pso_params['c1_config']
    c2_start, c2_end = pso_params['c2_config']; Pd_start, Pd_end = pso_params['Pd_config']
    Pl_start, Pl_end = pso_params['Pl_config']; Pr_start, Pr_end = pso_params['Pr_config']
    
    CONVERGENCE_THRESHOLD_GEN = pso_params['convergence_gen']
    TIME_LIMIT_SEC = pso_params['time_limit_sec']

    # 乱数のシードをリセット（2つのPSOの初期条件を揃えるため）
    # 注意: これでもNumPyの乱数状態は異なる可能性がある
    random.seed(int(pso_start_time)) 
    np.random.seed(int(pso_start_time))

    swarms = np.random.uniform(1, 20, (num_par, num_nodes)); velocities = np.zeros_like(swarms)
    pBests = np.copy(swarms); pBests_fitness = np.full(num_par, -float('inf'))
    gBest = swarms[0]; gBest_fitness = -float('inf')
    gBest_feasible_bn = -1; gBest_feasible_path = None
    
    stagnation_counter = 0
    last_best_bn = -1.0 
    terminated_reason = "max_gen"
    final_gen = num_gen 

    for i in range(num_gen): # 世代ループ
        current_pso_time = time.time() - pso_start_time
        
        if current_pso_time > TIME_LIMIT_SEC:
            if gBest_feasible_path is None:
                terminated_reason = "timeout_no_solution"
            else:
                terminated_reason = "timeout_with_solution"
            final_gen = i
            break # タイムアウトループを抜ける

        progress = i / num_gen
        w = w_start - (w_start - w_end) * progress; c1 = c1_start - (c1_start - c1_end) * progress
        c2 = c2_start + (c2_end - c2_start) * progress; P_d = Pd_start + (Pd_end - Pd_start) * progress
        P_l = Pl_start + (Pl_end - Pl_start) * progress; P_r = Pr_start + (Pr_end - Pr_start) * progress
        
        # ★★★ 変更点: 'delayed' モードのための「i-1」世代のpBest/gBestをコピー ★★★
        # (sim2131.py のロジックを模倣)
        if reference_mode == 'delayed':
            pBests_to_use = np.copy(pBests)
            gBest_to_use = np.copy(gBest)
        
        current_fitness = np.zeros(num_par)
        
        for j in range(num_par): # 粒子ループ
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
                gBest_feasible_path = path
                
                current_pso_time_on_update = time.time() - pso_start_time
                pso_progress_history.append((current_pso_time_on_update, gBest_feasible_bn))

        # --- pBest と gBest (fitness基準) の更新 ---
        # (この更新ロジックは watase_24.py と同一)
        update_indices = current_fitness > pBests_fitness
        pBests[update_indices] = swarms[update_indices]
        pBests_fitness[update_indices] = current_fitness[update_indices]
        current_best_idx = np.argmax(current_fitness)
        if current_fitness[current_best_idx] > gBest_fitness:
            gBest_fitness = current_fitness[current_best_idx]
            gBest = swarms[current_best_idx]
        
        # ★★★ 変更点: 'immediate' モードのためのpBest/gBestをセット ★★★
        if reference_mode == 'immediate':
            pBests_to_use = pBests
            gBest_to_use = gBest

        # --- 速度と位置の更新 ---
        r1, r2 = np.random.rand(2, num_par, 1)
        
        # ★★★ 変更点: 参照モードに応じて使用する pBest/gBest を切り替え ★★★
        velocities = w * velocities + c1 * r1 * (pBests_to_use - swarms) + c2 * r2 * (gBest_to_use.reshape(1, -1) - swarms)
        swarms += velocities

        # (収束判定ロジックは watase_24.py と同一)
        if gBest_feasible_bn > -1:
            if gBest_feasible_bn > last_best_bn:
                stagnation_counter = 0 
                last_best_bn = gBest_feasible_bn
            else:
                stagnation_counter += 1 
            
            if stagnation_counter >= CONVERGENCE_THRESHOLD_GEN:
                terminated_reason = "convergence"
                final_gen = i + 1 
                break # 収束ループを抜ける
    
    # --- ループ終了後の処理 ---
    pso_time = time.time() - pso_start_time
    pso_bn, pso_d, pso_l, pso_r = calculate_path_attributes_4d(Graph, gBest_feasible_path) if gBest_feasible_path else (-1,-1,-1,-1)

    # 8個のタプルと履歴リストを返す
    pso_results = (pso_bn, pso_time, gBest_feasible_path is not None,
                   pso_d, pso_l, pso_r,
                   final_gen, terminated_reason)
    
    return pso_results, pso_progress_history


def simulation(Graph, src, dst, constraints, pso_params):
    """
    厳密解法、即時参照PSO、遅延参照PSO の3つを実行する
    """
    
    # --- 1. 厳密解法 (LC) の実行 ---
    try: 
        min_delay_path_dijkstra = nx.dijkstra_path(Graph, source=src, target=dst, weight='delay')
        _, min_delay, _, _ = calculate_path_attributes_4d(Graph, min_delay_path_dijkstra)
    except nx.NetworkXNoPath: 
        print("Error: No path exists between source and target.")
        return None, None, None # 3つのアルゴリズムの結果をNoneで返す
        
    max_delay = min_delay * constraints['delay_multiplier']
    max_loss = constraints['loss_constraint']
    min_rel = constraints['reliability_constraint']
    
    print("  (1/3) 厳密解法 (LC) を実行中...")
    lc_start_time = time.time()
    opt_path, opt_bn, _, _, _, _ = find_optimal_path_by_label_correcting_4d(Graph, src, dst, max_delay, max_loss, min_rel)
    lc_time = time.time() - lc_start_time
    lc_results = (opt_bn, lc_time)
    
    # (制約条件を2つのPSOで共通化)
    pso_shared_constraints = (max_delay, max_loss, min_rel)

    # --- 2. 即時参照PSO (watase_24) の実行 ---
    print("  (2/3) 即時参照PSO (Immediate) を実行中...")
    pso_imm_results, pso_imm_history = _run_pso_loop(
        Graph, src, dst, constraints, pso_params, 
        *pso_shared_constraints, reference_mode='immediate'
    )
    
    # --- 3. 遅延参照PSO (sim2131) の実行 ---
    print("  (3/3) 遅延参照PSO (Delayed) を実行中...")
    pso_del_results, pso_del_history = _run_pso_loop(
        Graph, src, dst, constraints, pso_params, 
        *pso_shared_constraints, reference_mode='delayed'
    )

    # 3つの結果をタプルで返す
    return (lc_results, pso_imm_results, pso_del_results, pso_imm_history, pso_del_history)


if __name__ == '__main__':
    # (パラメータ設定は watase_24.py と同一)
    num_simulation = 5
    node_counts = [1000]
    graph_types = ['random']
    constraints = {'delay_multiplier': 20.0, 'loss_constraint': 1.0, 'reliability_constraint': 0.01}
    
    pso_params = {
        'num_par': 10, 'num_gen': 1000,
        'convergence_gen': 1000,
        'time_limit_sec': 1200, 
        'w_config': (0.9, 0.4), 'c1_config': (2.5, 0.5), 'c2_config': (0.5, 2.5),
        'Pd_config': (0.1, 1.0), 'Pl_config': (100, 1000), 'Pr_config': (1000, 10000)
    }

    # --- CSVファイルの準備 ---
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = savef.create_dir(dir_name="adaptive_termination_results_4D_FIXED")
    
    # ★★★ 変更点: ファイル名を 'comparison' に変更 ★★★
    progress_log_filename = result_path + f'/{run_timestamp}_4criteria_comparison_progress_log.csv'
    summary_filename = result_path + f'/{run_timestamp}_4criteria_comparison_summary.csv'
    
    # 時系列ログ (progress_log.csv) のヘッダー
    try:
        with open(progress_log_filename, 'w', newline='') as f_log:
            log_writer = csv.writer(f_log)
            log_writer.writerow(['Num_Nodes', 'Graph_Type', 'Iter', 'Algorithm', 'Time', 'Bottleneck'])
    except IOError as e:
        print(f"時系列ログファイルの作成に失敗しました: {e}")
        exit()
        
    # サマリー (summary.csv) のヘッダー
    summary_header = [
        'Num_Nodes', 'Graph_Type', 'Iter', 
        'Optimal_BN', 'Optimal_Time',
        'PSO_Imm_BN', 'PSO_Imm_Time', 'PSO_Imm_Feasible', 'PSO_Imm_Delay', 'PSO_Imm_Loss', 'PSO_Imm_Reliability', 'PSO_Imm_Final_Gen', 'PSO_Imm_Termination_Reason',
        'PSO_Del_BN', 'PSO_Del_Time', 'PSO_Del_Feasible', 'PSO_Del_Delay', 'PSO_Del_Loss', 'PSO_Del_Reliability', 'PSO_Del_Final_Gen', 'PSO_Del_Termination_Reason'
    ]
    try:
        with open(summary_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(summary_header)
    except IOError as e:
        print(f"サマリーファイルの作成に失敗しました: {e}")
        exit()
        
    # --- シミュレーション実行 ---
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

                # ★★★ 変更点: 3つの結果を受け取る ★★★
                (lc_res, pso_imm_res, pso_del_res, 
                 pso_imm_hist, pso_del_hist) = simulation(Graph, src, dst, constraints, pso_params)

                if lc_res is None: # パスが見つからなかった場合
                    continue

                # --- 1. 時系列ログ (progress_log.csv) への追記 ---
                try:
                    with open(progress_log_filename, 'a', newline='') as f_log:
                        log_writer = csv.writer(f_log)
                        
                        # 厳密解法
                        if lc_res[0] > -1:
                            log_writer.writerow([num_nodes, graph_type, i + 1, 'Exact', lc_res[1], lc_res[0]])
                        
                        # 即時参照PSO
                        for (log_time, log_bn) in pso_imm_hist:
                            log_writer.writerow([num_nodes, graph_type, i + 1, 'PSO_Immediate', log_time, log_bn])
                        
                        # 遅延参照PSO
                        for (log_time, log_bn) in pso_del_hist:
                            log_writer.writerow([num_nodes, graph_type, i + 1, 'PSO_Delayed', log_time, log_bn])
                except IOError as e:
                    print(f"時系列ログファイルへの書き込みエラー: {e}")

                # --- 2. サマリー (summary.csv) への追記 ---
                try:
                    with open(summary_filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        # [Num_Nodes, Graph_Type, Iter] + [LC_BN, LC_Time] + [PSO_Imm_Results (8)] + [PSO_Del_Results (8)]
                        row_data = ([num_nodes, graph_type, i + 1] + 
                                    list(lc_res) + 
                                    list(pso_imm_res) + 
                                    list(pso_del_res))
                        writer.writerow(row_data)
                except IOError as e:
                    print(f"サマリーファイルへの書き込みエラー: {e}")

    print(f"\n実験サマリーが '{summary_filename}' に保存されました。")
    print(f"時系列ログが '{progress_log_filename}' に保存されました。")