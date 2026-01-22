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

# --- 1. 最適化された PathEncode (リスト生成の無駄を排除) ---
def PathEncode(Particle, Graph, src, dst):
    path = [src]
    current_node = src
    visited = {src}
    
    # 事前に長さを取得（これは維持）
    limit_len = len(Particle)
    
    while current_node != dst:
        # ★修正: Pythonのforループは遅いので、リスト内包表記に戻す
        # filter処理(if n not in visited)がC言語レベルで走るため、こちらの方が速い
        neighbors = [n for n in Graph.neighbors(current_node) if n not in visited]
        
        if not neighbors: 
            return path, False
            
        best_neighbor = -1
        highest_prio = -1.0
        
        # ここは要素数が絞られた後のループなのでPythonでも十分速い
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
    # (厳密解法のコードは変更なし。heapqを使用しており十分最適化されています)
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

# --- 2. 新規追加: Jaccard係数に基づくパス類似度近傍 (行列演算版) ---
def get_jaccard_lbest(pBests, pBests_fitness, current_paths, num_nodes, k=5):
    """
    パス類似度（Jaccard係数）に基づいて各粒子のlBestを決定する。
    
    Args:
        pBests (np.array): pBest位置 (num_par, dim)
        pBests_fitness (np.array): pBest評価値 (num_par,)
        current_paths (list of list): 各粒子が現在生成している経路 [[0, 1, 5], [0, 2, ...], ...]
        num_nodes (int): ノード総数
        k (int): 自分を含む近傍粒子の数
    """
    num_par = len(pBests)
    
    # 1. パスを行列化 (Particle x Node の0/1行列)
    # 通ったノードには1、通ってないノードには0
    path_matrix = np.zeros((num_par, num_nodes), dtype=np.float32) # floatの方が内積計算で有利
    for i, path in enumerate(current_paths):
        if path: # パスが存在する場合
            path_matrix[i, path] = 1.0
            
    # 2. Jaccard係数の高速計算 (行列演算)
    # Intersection (共通ノード数) = M dot M.T
    intersection = np.dot(path_matrix, path_matrix.T)
    
    # Union (和集合ノード数) = |A| + |B| - |A cap B|
    # 各パスの長さ（ノード数）
    path_lengths = np.sum(path_matrix, axis=1)
    # broadcasting: (N,1) + (1,N)
    union = path_lengths[:, np.newaxis] + path_lengths[np.newaxis, :] - intersection
    
    # ゼロ除算回避（パス生成失敗した粒子同士など）
    union[union == 0] = 1.0
    
    # Jaccard行列
    jaccard_matrix = intersection / union
    
    # 自分自身(対角成分)は類似度1.0だが、argmax等で邪魔しないように扱われる
    
    # 3. 近傍k個の選択
    # 類似度が高い順にk個選ぶ (大きい順なので -jaccard をソートするか、argpartitionの仕様に合わせる)
    # argpartitionは「小さい順」なので、符号反転して上位k個を取る
    nearest_indices = np.argpartition(-jaccard_matrix, kth=k, axis=1)[:, :k]
    
    # 4. lBestの決定
    lBest_matrix = np.zeros_like(pBests)
    for i in range(num_par):
        neighbors = nearest_indices[i]
        neighbor_fitness = pBests_fitness[neighbors]
        best_local_idx = np.argmax(neighbor_fitness)
        best_global_idx = neighbors[best_local_idx]
        lBest_matrix[i] = pBests[best_global_idx]
        
    return lBest_matrix


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

    # --- PSOの実行 ---
    pso_start_time = time.time()
    pso_generation_history = [] 

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
    last_best_bn = -1.0 
    terminated_reason = "max_gen"
    final_gen = num_gen 

    for i in range(num_gen):
        current_pso_time = time.time() - pso_start_time
        if current_pso_time > TIME_LIMIT_SEC:
            terminated_reason = "timeout" if gBest_feasible_path is None else "timeout_with_solution"
            final_gen = i 
            break

        progress = i / num_gen
        w = w_start - (w_start - w_end) * progress; c1 = c1_start - (c1_start - c1_end) * progress
        c2 = c2_start + (c2_end - c2_start) * progress; P_d = Pd_start + (Pd_end - Pd_start) * progress
        P_l = Pl_start + (Pl_end - Pl_start) * progress; P_r = Pr_start + (Pr_end - Pr_start) * progress
        
        current_fitness = np.zeros(num_par)
        
        # Jaccard計算用に全粒子のパスを保存するリスト
        current_paths_list = []
        
        for j in range(num_par):
            # 最適化されたPathEncodeを使用
            path, is_valid = PathEncode(swarms[j], Graph, src, dst)
            current_paths_list.append(path) # パスを保存
            
            if not is_valid: 
                current_fitness[j] = -1.0
                continue
            
            bn, d, l, r = calculate_path_attributes_4d(Graph, path)
            fitness = bn; penalty = 0
            
            # 論理的ペナルティ計算
            if d > max_delay: penalty += P_d * (d - max_delay)
            if l > max_loss: penalty += P_l * (l - max_loss)
            if r < min_rel: penalty += P_r * (min_rel - r)
            
            fitness -= penalty
            current_fitness[j] = fitness
            is_feasible = (d <= max_delay and l <= max_loss and r >= min_rel)
            
            if is_feasible and bn > gBest_feasible_bn:
                gBest_feasible_bn = bn
                gBest_feasible_path = path

        # pBest, gBest 更新
        update_indices = current_fitness > pBests_fitness
        pBests[update_indices] = swarms[update_indices]; pBests_fitness[update_indices] = current_fitness[update_indices]
        
        # gBest更新 (Fitnessベース)
        current_best_idx = np.argmax(current_fitness)
        if current_fitness[current_best_idx] > gBest_fitness:
            gBest_fitness = current_fitness[current_best_idx]; gBest = swarms[current_best_idx]
        
        # --- ★★★ Step 2: Jaccard係数による近傍計算 ★★★ ---
        # k=20程度に増やしてみる（似た者同士の中なら広く探しても安全だから）
        lBest_matrix = get_jaccard_lbest(pBests, pBests_fitness, current_paths_list, num_nodes, k=20)
        
        r1, r2 = np.random.rand(2, num_par, 1)
        velocities = w * velocities + c1 * r1 * (pBests - swarms) + c2 * r2 * (lBest_matrix - swarms)
        swarms += velocities

        # 収束判定
        if gBest_feasible_bn > -1:
            if gBest_feasible_bn > last_best_bn:
                stagnation_counter = 0; last_best_bn = gBest_feasible_bn 
            else:
                stagnation_counter += 1 
            
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
    num_simulation = 3 
    node_counts = [1000] 
    graph_types = ['random']
    constraints = {'delay_multiplier': 3.0, 'loss_constraint': 0.1, 'reliability_constraint': 0.95}
    
    # --- 3. 論理的ペナルティに基づいたパラメータ設定 ---
    pso_params = {
        'num_par': 100, 'num_gen': 100,
        'convergence_gen': 100,
        'time_limit_sec': 1200, 
        'w_config': (0.9, 0.5), 'c1_config': (2.5, 0.5), 'c2_config': (0.5, 2.5),
        # 2. ペナルティ
        'Pd_config': (0.1, 1.0), 
        'Pl_config': (100.0, 500.0),  
        'Pr_config': (500.0, 1000.0)
    }

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = savef.create_dir(dir_name="adaptive_termination_results_Step2_Jaccard")
    
    generation_log_filename = result_path + f'/{run_timestamp}_jaccard_generation_log.csv'
    try:
        with open(generation_log_filename, 'w', newline='') as f_log:
            log_writer = csv.writer(f_log)
            log_writer.writerow(['Num_Nodes', 'Graph_Type', 'Iter', 'Generation', 'Time', 'Bottleneck'])
    except IOError as e: print(f"File Error: {e}"); exit()
        
    summary_filename = result_path + f'/{run_timestamp}_jaccard_summary.csv'
    try:
        with open(summary_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Num_Nodes', 'Graph_Type', 'Iter', 'Optimal_BN', 'Optimal_Time',
                             'PSO_Final_BN', 'PSO_Final_Time', 'PSO_Feasible', 'PSO_Final_Delay', 'PSO_Final_Loss', 'PSO_Final_Reliability',
                             'PSO_Final_Gen', 'PSO_Termination_Reason'])
    except IOError as e: print(f"File Error: {e}"); exit()

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