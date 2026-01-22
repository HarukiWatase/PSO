# watase_21_pso_progress_logging_FIXED.py
# (ベース: watase_20_pso_adaptive_termination_FIXED.py)

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
This script implements a PSO with adaptive termination conditions.
★★★ The timeout logic has been corrected to apply universally. ★★★

[変更点 (watase_21)]
- グラフ描画のため、PSOが実行可能解を更新するたびに
  (時刻, ボトルネック値) を記録する機能を追加。
- 最終結果の「サマリーCSV」とは別に、「時系列ログCSV」を
  新規に出力するよう変更。
- アルゴリズムの探索ロジック自体には変更なし。
"""

# (PathEncode, calculate_path_attributes_4d, find_optimal_path_by_label_correcting_4d は変更ありません)
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

def simulation(Graph, src, dst, constraints, pso_params):
    # (厳密解法の実行部分は変更なし)
    try: 
        min_delay_path_dijkstra = nx.dijkstra_path(Graph, source=src, target=dst, weight='delay')
        _, min_delay, _, _ = calculate_path_attributes_4d(Graph, min_delay_path_dijkstra)
    except nx.NetworkXNoPath: 
        print("Error: No path exists between source and target.")
        return (None,) * 11 
        
    max_delay = min_delay * constraints['delay_multiplier']
    max_loss = constraints['loss_constraint']
    min_rel = constraints['reliability_constraint']
    lc_start_time = time.time()
    opt_path, opt_bn, _, _, _, _ = find_optimal_path_by_label_correcting_4d(Graph, src, dst, max_delay, max_loss, min_rel)
    lc_time = time.time() - lc_start_time

    # --- PSOの実行 ---
    pso_start_time = time.time()
    pso_progress_history = [] 

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
    # ★★★ 収束判定のためだけに使う「前の世代の最良解」 ★★★
    last_best_bn = -1.0 
    terminated_reason = "max_gen"
    final_gen = num_gen 

    for i in range(num_gen): # 世代ループ
        current_pso_time = time.time() - pso_start_time
        
        # (タイムアウト判定ロジックは変更なし)
        if current_pso_time > TIME_LIMIT_SEC:
            if gBest_feasible_path is None:
                terminated_reason = "timeout_no_solution"
                print(f"INFO: {TIME_LIMIT_SEC}秒以内に実行可能解が見つからなかったため、探索を打ち切ります。")
            else:
                terminated_reason = "timeout_with_solution"
                print(f"INFO: {TIME_LIMIT_SEC}秒の制限時間に達したため、探索を打ち切ります。")
            final_gen = i # タイムアウトした世代を記録
            break

        # (パラメータ更新ロジックは変更なし)
        progress = i / num_gen
        w = w_start - (w_start - w_end) * progress; c1 = c1_start - (c1_start - c1_end) * progress
        c2 = c2_start + (c2_end - c2_start) * progress; P_d = Pd_start + (Pd_end - Pd_start) * progress
        P_l = Pl_start + (Pl_end - Pl_start) * progress; P_r = Pr_start + (Pr_end - Pr_start) * progress
        
        current_fitness = np.zeros(num_par)
        
        for j in range(num_par): # 粒子ループ
            path, is_valid = PathEncode(swarms[j], Graph, src, dst)
            if not is_valid: current_fitness[j] = -1.0; continue
            
            bn, d, l, r = calculate_path_attributes_4d(Graph, path)
            
            # (fitness計算ロジックは変更なし)
            fitness = bn; penalty = 0
            if d > max_delay: penalty += P_d * (d - max_delay)
            if l > max_loss: penalty += P_l * (l - max_loss)
            if r < min_rel: penalty += P_r * (min_rel - r)
            fitness -= penalty
            current_fitness[j] = fitness
            
            is_feasible = (d <= max_delay and l <= max_loss and r >= min_rel)
            
            # ▼▼▼ ★★★ 修正点: ログ記録をここに移動 ★★★ ▼▼▼
            # 粒子j が、現在の「全体最良」よりも良い解を見つけた
            if is_feasible and bn > gBest_feasible_bn:
                gBest_feasible_bn = bn # 全体最良を更新
                gBest_feasible_path = path
                
                # ★★★ 更新した瞬間の時刻と解を記録 ★★★
                current_pso_time_on_update = time.time() - pso_start_time
                pso_progress_history.append((current_pso_time_on_update, gBest_feasible_bn))
            # ▲▲▲ ★★★ 修正ここまで ★★★ ▲▲▲

        # (pBest, gBest(fitness基準) の更新ロジックは変更なし)
        update_indices = current_fitness > pBests_fitness
        pBests[update_indices] = swarms[update_indices]; pBests_fitness[update_indices] = current_fitness[update_indices]
        current_best_idx = np.argmax(current_fitness)
        if current_fitness[current_best_idx] > gBest_fitness:
            gBest_fitness = current_fitness[current_best_idx]; gBest = swarms[current_best_idx]
        
        # (速度と位置の更新ロジックは変更なし)
        r1, r2 = np.random.rand(2, num_par, 1)
        velocities = w * velocities + c1 * r1 * (pBests - swarms) + c2 * r2 * (gBest.reshape(1, -1) - swarms)
        swarms += velocities

        # ▼▼▼ 収束判定ロジック (ログ記録コードは削除) ▼▼▼
        # (gBest_feasible_path is not None は gBest_feasible_bn > -1 と同義)
        if gBest_feasible_bn > -1:
            
            # (世代i終了時点の最良解) gBest_feasible_bn が 
            # (世代i-1までの最良解) last_best_bn よりも
            # 良い場合 (>) のみ、ブロックの中に入る
            if gBest_feasible_bn > last_best_bn:
                stagnation_counter = 0 
                last_best_bn = gBest_feasible_bn # 収束判定用の最良解を更新
            else:
                # この世代では解の更新がなかった
                stagnation_counter += 1 
            
            # (収束判定は変更なし)
            if stagnation_counter >= CONVERGENCE_THRESHOLD_GEN:
                terminated_reason = "convergence"
                final_gen = i + 1 # 収束した世代を記録
                print(f"INFO: {CONVERGENCE_THRESHOLD_GEN}世代にわたり解が改善されなかったため、収束したと見なし探索を打ち切ります。")
                break
    
    pso_time = time.time() - pso_start_time
    
    # (最終解の計算は変更なし)
    pso_bn, pso_d, pso_l, pso_r = calculate_path_attributes_4d(Graph, gBest_feasible_path) if gBest_feasible_path else (-1,-1,-1,-1)

    return (opt_bn, lc_time,
            pso_bn, pso_time, gBest_feasible_path is not None,
            pso_d, pso_l, pso_r,
            final_gen, terminated_reason,
            pso_progress_history) # ★★★ 履歴を返す ★★★

if __name__ == '__main__':
    summary_results = []
    
    # (main関数内のパラメータは、watase_20.py のものをそのまま使用)
    num_simulation = 100
    node_counts = [100]
    graph_types = ['random']
    constraints = {'delay_multiplier': 20.0, 'loss_constraint': 1.0, 'reliability_constraint': 0.01}
    
    pso_params = {
        'num_par': 10, 'num_gen': 1000,
        'convergence_gen': 1000,
        'time_limit_sec': 1200, # 20分
        'w_config': (0.9, 0.4), 'c1_config': (2.5, 0.5), 'c2_config': (0.5, 2.5),
        'Pd_config': (0.1, 1.0), 'Pl_config': (100, 1000), 'Pr_config': (1000, 10000)
    }

    # --- CSVファイル（時系列ログ）の準備 ---
    # ★★★ 変更点: グラフ描画用の時系列ログファイルを追加 ★★★
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = savef.create_dir(dir_name="adaptive_termination_results_4D_FIXED")
    progress_log_filename = result_path + f'/{run_timestamp}_4criteria_progress_log.csv'
    
    try:
        with open(progress_log_filename, 'w', newline='') as f_log:
            log_writer = csv.writer(f_log)
            log_writer.writerow(['Num_Nodes', 'Graph_Type', 'Iter', 'Algorithm', 'Time', 'Bottleneck'])
    except IOError as e:
        print(f"時系列ログファイルの作成に失敗しました: {e}")
        exit()
    # ★★★ 変更ここまで ★★★

    for num_nodes in node_counts:
        for graph_type in graph_types:
            for i in range(num_simulation):
                print(f'\n--- Sim (N:{num_nodes}, G:{graph_type}, iter:{i+1}) ---')
                Graph = getattr(create_graph, f"{graph_type}_graph")(num_node=num_nodes)
                if not nx.is_connected(Graph):
                    largest_cc = max(nx.connected_components(Graph), key=len)
                    if len(largest_cc) < 2: 
                        print("Skip: Graph CC < 2")
                        continue
                    node_list = list(largest_cc)
                else: 
                    node_list = list(Graph.nodes())
                if len(node_list) < 2: 
                    print("Skip: Node list < 2")
                    continue
                src, dst = random.sample(node_list, 2)

                (opt_bn, lc_time,
                 pso_bn, pso_time, pso_feasible, 
                 pso_d, pso_l, pso_r,
                 pso_final_gen, pso_term_reason,
                 pso_progress_history) = simulation(Graph, src, dst, constraints, pso_params) # ★★★ 戻り値追加 ★★★

                # ★★★ 変更点: 時系列ログファイルへの追記 ★★★
                try:
                    with open(progress_log_filename, 'a', newline='') as f_log:
                        log_writer = csv.writer(f_log)
                        # 厳密解法の結果 (実行可能解が見つかった場合のみ)
                        if opt_bn is not None and opt_bn > -1:
                            log_writer.writerow([num_nodes, graph_type, i + 1, 'Exact', lc_time, opt_bn])
                        # PSOの実行履歴 (実行可能解が見つかった場合のみ)
                        for (log_time, log_bn) in pso_progress_history:
                            log_writer.writerow([num_nodes, graph_type, i + 1, 'PSO', log_time, log_bn])
                except IOError as e:
                    print(f"時系列ログファイルへの書き込みエラー: {e}")
                # ★★★ 変更ここまで ★★★

                # サマリーCSV用のデータ追加 (変更なし)
                summary_results.append([
                    num_nodes, graph_type, i + 1, opt_bn, lc_time,
                    pso_bn, pso_time, pso_feasible, pso_d, pso_l, pso_r,
                    pso_final_gen, pso_term_reason
                ])

    # --- CSVファイル（サマリー）への書き出し ---
    # ★★★ 変更点: サマリー用のファイル名を設定 ★★★
    summary_filename = result_path + f'/{run_timestamp}_4criteria_adaptive_FIXED_summary.csv'
    try:
        with open(summary_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Num_Nodes', 'Graph_Type', 'Iter', 'Optimal_BN', 'Optimal_Time',
                             'PSO_BN', 'PSO_Time', 'PSO_Feasible', 'PSO_Delay', 'PSO_Loss', 'PSO_Reliability',
                             'PSO_Final_Gen', 'PSO_Termination_Reason'])
            writer.writerows(summary_results)
        
        print(f"\n実験サマリーが '{summary_filename}' に保存されました。")
        print(f"時系列ログが '{progress_log_filename}' に保存されました。") # ★★★ 追記 ★★★
    except IOError as e:
        print(f"サマリーファイルの書き込みに失敗しました: {e}")