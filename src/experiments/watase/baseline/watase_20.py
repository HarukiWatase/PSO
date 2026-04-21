# watase_20_pso_adaptive_termination.py

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
This script implements a PSO with adaptive termination conditions:
1. Convergence-based termination after a feasible solution is found.
2. Time-based termination if no feasible solution is found.
"""

# (PathEncode, calculate_path_attributes_4d, find_optimal_path_by_label_correcting_4d は変更ありません)
def PathEncode(Particle, Graph, src, dst):
    path = [src]
    current_node = src
    visited = {src}
    while current_node != dst:
        neighbors = [n for n in Graph.neighbors(current_node) if n not in visited]
        if not neighbors: return path, False
        highest_prio = -1; next_node = -1
        for neighbor in neighbors:
            if Particle[neighbor] > highest_prio:
                highest_prio = Particle[neighbor]; next_node = neighbor
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
    initial_label = (0.0, float('inf'), 0.0, 0.0)
    labels[source].append(initial_label); pred[source][initial_label] = (None, None)
    heappush(pq, (-initial_label[1], initial_label[0], initial_label[2], initial_label[3], source, initial_label))
    min_reliability_cost = -math.log(min_reliability) if min_reliability > 0 else float('inf')
    while pq:
        neg_bottle, d_curr, l_curr, r_curr, u, label_curr = heappop(pq)
        if label_curr not in labels[u]: continue
        for v in G.neighbors(u):
            edge = G.get_edge_data(u, v)
            d_new = d_curr + edge.get("delay", 0); b_new = min(-neg_bottle, edge.get("weight", 1))
            l_new = l_curr + edge.get("loss_log_cost", 0); r_new = r_curr + edge.get("reliability_cost", 0)
            if d_new > max_delay: continue
            if l_new > -math.log(1 - max_loss_rate): continue
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
    try: min_delay = nx.dijkstra_path_length(Graph, source=src, target=dst, weight='delay')
    except nx.NetworkXNoPath: return (None,) * 10
    max_delay = min_delay * constraints['delay_multiplier']
    max_loss = constraints['loss_constraint']
    min_rel = constraints['reliability_constraint']
    lc_start_time = time.time()
    opt_path, opt_bn, _, _, _, _ = find_optimal_path_by_label_correcting_4d(Graph, src, dst, max_delay, max_loss, min_rel)
    lc_time = time.time() - lc_start_time

    # --- PSOの実行 ---
    pso_start_time = time.time()
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

    for i in range(num_gen):
        if gBest_feasible_path is None and (time.time() - pso_start_time) > TIME_LIMIT_SEC:
            terminated_reason = "timeout"
            print(f"INFO: {TIME_LIMIT_SEC}秒以内に実行可能解が見つからなかったため、探索を打ち切ります。")
            break

        progress = i / num_gen
        w = w_start - (w_start - w_end) * progress; c1 = c1_start - (c1_start - c1_end) * progress
        c2 = c2_start + (c2_end - c2_start) * progress; P_d = Pd_start + (Pd_end - Pd_start) * progress
        P_l = Pl_start + (Pl_end - Pl_start) * progress; P_r = Pr_start + (Pr_end - Pr_start) * progress
        
        current_fitness = np.zeros(num_par)
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
                gBest_feasible_bn = bn; gBest_feasible_path = path

        update_indices = current_fitness > pBests_fitness
        pBests[update_indices] = swarms[update_indices]; pBests_fitness[update_indices] = current_fitness[update_indices]
        current_best_idx = np.argmax(current_fitness)
        if current_fitness[current_best_idx] > gBest_fitness:
            gBest_fitness = current_fitness[current_best_idx]; gBest = swarms[current_best_idx]
        
        r1, r2 = np.random.rand(2, num_par, 1)
        velocities = w * velocities + c1 * r1 * (pBests - swarms) + c2 * r2 * (gBest.reshape(1, -1) - swarms)
        swarms += velocities

        if gBest_feasible_path is not None:
            if gBest_feasible_bn > last_best_bn:
                stagnation_counter = 0
                last_best_bn = gBest_feasible_bn
            else:
                stagnation_counter += 1
            
            if stagnation_counter >= CONVERGENCE_THRESHOLD_GEN:
                terminated_reason = "convergence"
                print(f"INFO: {CONVERGENCE_THRESHOLD_GEN}世代にわたり解が改善されなかったため、収束したと見なし探索を打ち切ります。")
                break
    
    pso_time = time.time() - pso_start_time
    final_gen = i + 1
    pso_bn, pso_d, pso_l, pso_r = calculate_path_attributes_4d(Graph, gBest_feasible_path) if gBest_feasible_path else (-1,-1,-1,-1)

    return (opt_bn, lc_time,
            pso_bn, pso_time, gBest_feasible_path is not None,
            pso_d, pso_l, pso_r,
            final_gen, terminated_reason)

if __name__ == '__main__':
    summary_results = []
    num_simulation = 10
    node_counts = [3200,3400,3600,3800,4000]
    graph_types = ['random']
    constraints = {'delay_multiplier': 3.0, 'loss_constraint': 0.1, 'reliability_constraint': 0.95}
    
    pso_params = {
        'num_par': 200, 'num_gen': 500,
        'convergence_gen': 30,
        'time_limit_sec': 1200,
        'w_config': (0.9, 0.4), 'c1_config': (2.5, 0.5), 'c2_config': (0.5, 2.5),
        'Pd_config': (0.1, 1.0), 'Pl_config': (100, 1000), 'Pr_config': (1000, 10000)
    }

    for num_nodes in node_counts:
        for graph_type in graph_types:
            for i in range(num_simulation):
                print(f'\n--- Sim (N:{num_nodes}, G:{graph_type}, iter:{i+1}) ---')
                Graph = getattr(create_graph, f"{graph_type}_graph")(num_node=num_nodes)
                if not nx.is_connected(Graph):
                    largest_cc = max(nx.connected_components(Graph), key=len)
                    if len(largest_cc) < 2: continue
                    node_list = list(largest_cc)
                else: node_list = list(Graph.nodes())
                src, dst = random.sample(node_list, 2)

                (opt_bn, lc_time,
                 pso_bn, pso_time, pso_feasible, 
                 pso_d, pso_l, pso_r,
                 pso_final_gen, pso_term_reason) = simulation(Graph, src, dst, constraints, pso_params)

                summary_results.append([
                    num_nodes, graph_type, i + 1, opt_bn, lc_time,
                    pso_bn, pso_time, pso_feasible, pso_d, pso_l, pso_r,
                    pso_final_gen, pso_term_reason
                ])

    # --- CSVファイルへの書き出し ---
    # ★★★ 修正点: savef.create_dir を呼び出して result_path を定義 ★★★
    result_path = savef.create_dir(dir_name="adaptive_termination_results")
    filename = result_path + f'/{datetime.now().strftime("%Y%m%d_%H%M%S")}_adaptive_termination.csv'
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Num_Nodes', 'Graph_Type', 'Iter', 'Optimal_BN', 'Optimal_Time',
                         'PSO_BN', 'PSO_Time', 'PSO_Feasible', 'PSO_Delay', 'PSO_Loss', 'PSO_Reliability',
                         'PSO_Final_Gen', 'PSO_Termination_Reason'])
        writer.writerows(summary_results)
    print(f"\n実験結果が '{filename}' に保存されました。")