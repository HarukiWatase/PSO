# watase_23_2criteria_adaptive_pso.py

import networkx as nx
import numpy as np
import create_graph # ★★★ 2基準グラフ生成モジュールをインポート ★★★
from datetime import datetime
import csv
import savef # Assumes savef module exists
from heapq import heappop, heappush
import time
import random

"""
This script implements a PSO for the 2-criteria MCOP 
(Maximize Bottleneck, Constraint: Delay) with adaptive termination conditions.
It compares the results with the 2-criteria label-correcting algorithm.
"""

# --- PSO Helper Functions ---
def PathEncode(Particle, Graph, src, dst):
    """Decodes a particle's priority vector into a path."""
    path = [src]
    current_node = src
    visited = {src}
    # Use integer indices for Particle array access
    # Assuming nodes are 0..N-1 based on create_graph.py
    particle_values = Particle 

    while current_node != dst:
        neighbors = [n for n in Graph.neighbors(current_node) if n not in visited]
        if not neighbors: return path, False
        highest_prio = -1
        next_node = -1
        for neighbor in neighbors:
            # Get priority value for the neighbor node
            # Ensure neighbor index is valid for particle_values array
            if neighbor < len(particle_values):
                neighbor_prio = particle_values[neighbor]
                if neighbor_prio > highest_prio:
                    highest_prio = neighbor_prio
                    next_node = neighbor
            # else: # Handle case where neighbor index might be out of bounds if nodes aren't 0..N-1
                # print(f"Warning: Neighbor node {neighbor} out of bounds for Particle array.") # Optional warning
                # pass # Or handle error appropriately
        if next_node == -1: return path, False
        current_node = next_node
        path.append(current_node)
        visited.add(current_node)
    return path, path[-1] == dst


def calculate_path_attributes_2d(G, path):
    """Calculates bottleneck and delay for a path."""
    if not path or len(path) < 2: 
        return 0, float('inf') # Return 0 bottleneck, infinite delay
        
    bottleneck = float('inf')
    total_delay = 0
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if not G.has_edge(u, v): 
            return 0, float('inf') # Invalid path segment
            
        edge_data = G.edges[u, v]
        bottleneck = min(bottleneck, edge_data.get('weight', 0))
        total_delay += edge_data.get('delay', float('inf'))

    # If bottleneck remained infinite (e.g., path of length 1), return 0
    return bottleneck if bottleneck != float('inf') else 0, total_delay


# --- Exact Algorithm (2-Criteria Label-Correcting) ---
def find_optimal_path_by_label_correcting_2d(G, source, target, max_delay):
    """Finds the path with max bottleneck satisfying delay constraint."""
    labels = {node: [] for node in G.nodes()} 
    pred = {node: {} for node in G.nodes()} 
    pq = [] 
    initial_delay = 0.0; initial_bottle = float('inf')
    start_label = (initial_delay, initial_bottle)
    labels[source].append(start_label)
    pred[source][start_label] = (None, None) 
    heappush(pq, (-initial_bottle, initial_delay, source, start_label))

    while pq:
        neg_bottle, current_delay, u, current_label = heappop(pq)
        current_bottle = -neg_bottle
        if current_label not in labels[u]: continue

        for v in G.neighbors(u):
            edge_data = G.get_edge_data(u, v)
            edge_weight = edge_data.get('weight', 1) 
            edge_delay = edge_data.get('delay', 0)   
            new_delay = current_delay + edge_delay
            new_bottle = min(current_bottle, edge_weight)

            if new_delay > max_delay: continue
            new_label = (new_delay, new_bottle)
            
            is_dominated = False
            for d_exist, b_exist in labels[v]:
                if d_exist <= new_delay and b_exist >= new_bottle:
                    is_dominated = True; break
            if is_dominated: continue
            
            labels[v] = [(d, b) for d, b in labels[v] if not (d >= new_delay and b <= new_bottle)]
            labels[v].append(new_label)
            pred[v][new_label] = (u, current_label)
            heappush(pq, (-new_bottle, new_delay, v, new_label))

    final_labels_at_target = labels.get(target, [])
    if not final_labels_at_target: return None, -1, -1 
    best_bottle = -1; best_solution_label = None
    for d, b in final_labels_at_target:
        if d <= max_delay: 
            if b > best_bottle:
                best_bottle = b; best_solution_label = (d, b)
    if best_solution_label is None: return None, -1, -1 
    
    path = []; curr_node = target; curr_label = best_solution_label
    while curr_node is not None:
        path.append(curr_node); parent_node, parent_label = pred[curr_node][curr_label]
        curr_node = parent_node; curr_label = parent_label
    path.reverse()
    final_delay, final_bottle = best_solution_label
    return path, final_bottle, final_delay


def simulation(Graph, src, dst, constraints, pso_params):
    """Runs Label-Correcting and adaptive PSO for the 2-criteria problem."""
    print(f'Source->Target: {src}->{dst}')
    
    # --- Constraint Setup ---
    try:
        # Use Dijkstra to find the shortest delay path for setting the constraint
        min_delay_path_dijkstra = nx.dijkstra_path(Graph, source=src, target=dst, weight='delay')
        # ★★★ 修正点: calculate_path_attributes_2d を使用 ★★★
        _, min_delay = calculate_path_attributes_2d(Graph, min_delay_path_dijkstra) 
    except nx.NetworkXNoPath:
        print("Error: No path exists between source and target.")
        return (-1,) * 8 # 戻り値の数 (8つ) に合わせてタプルを返す

    max_delay = min_delay * constraints['delay_multiplier']
    print(f"Constraint - Delay: < {max_delay:.2f} (based on min delay {min_delay:.2f})")

    # --- 1. Exact Algorithm (Label-Correcting 2D) ---
    lc_start_time = time.time()
    opt_path, opt_bn, opt_d = find_optimal_path_by_label_correcting_2d(Graph, src, dst, max_delay)
    lc_time = time.time() - lc_start_time
    print(f"--- [Exact Algorithm Result] ---")
    if opt_path:
        print(f"Execution Time: {lc_time:.4f} sec, Found Path: BN={opt_bn:.2f}, D={opt_d:.2f}")
    else:
        print(f"Execution Time: {lc_time:.4f} sec, No feasible path found.")
        # If exact method finds no path, PSO likely won't either, but we still run it
        opt_bn, opt_d = -1, -1 

    # --- 2. PSO Execution (Adaptive Termination) ---
    pso_start_time = time.time()
    num_nodes = len(Graph.nodes())
    num_par, num_gen = pso_params['num_par'], pso_params['num_gen']
    w_start, w_end = pso_params['w_config']
    c1_start, c1_end = pso_params['c1_config']
    c2_start, c2_end = pso_params['c2_config']
    Pd_start, Pd_end = pso_params['Pd_config'] 
    CONVERGENCE_THRESHOLD_GEN = pso_params['convergence_gen']
    TIME_LIMIT_SEC = pso_params['time_limit_sec']

    # Initialization
    swarms = np.random.uniform(1, 20, (num_par, num_nodes))
    velocities = np.zeros_like(swarms)
    pBests = np.copy(swarms)
    pBests_fitness = np.full(num_par, -float('inf'))
    gBest = np.copy(swarms[0]) # Initialize gBest position
    gBest_fitness = -float('inf')
    gBest_feasible_bn = -1
    gBest_feasible_delay = -1
    gBest_feasible_path = None # Store the best feasible path itself

    # Adaptive termination variables
    stagnation_counter = 0
    last_best_bn = -1.0
    terminated_reason = "max_gen"
    final_gen = num_gen # Default final generation

    # Optimization Loop
    for i in range(num_gen):
        current_pso_time = time.time() - pso_start_time
        # Timeout check
        if gBest_feasible_path is None and current_pso_time > TIME_LIMIT_SEC:
            terminated_reason = "timeout"
            final_gen = i # Record generation where timeout occurred
            print(f"INFO: PSO timed out after {current_pso_time:.2f} sec without finding a feasible solution.")
            break

        # Update parameters
        progress = i / num_gen
        w = w_start - (w_start - w_end) * progress
        c1 = c1_start - (c1_start - c1_end) * progress
        c2 = c2_start + (c2_end - c2_start) * progress
        P_d = Pd_start + (Pd_end - Pd_start) * progress
        
        current_fitness = np.zeros(num_par)
        for j in range(num_par):
            path, is_valid = PathEncode(swarms[j], Graph, src, dst)
            if not is_valid: 
                current_fitness[j] = -1.0 # Use -1.0 as fitness for invalid path
                continue
            
            bn, delay = calculate_path_attributes_2d(Graph, path)
            
            # Fitness calculation (Maximize Bottleneck - Penalty)
            fitness = bn
            penalty = 0
            if delay > max_delay:
                penalty = P_d * (delay - max_delay)
            fitness -= penalty
            current_fitness[j] = fitness

            # Update best feasible solution
            is_feasible = (delay <= max_delay)
            if is_feasible and bn > gBest_feasible_bn:
                gBest_feasible_bn = bn
                gBest_feasible_delay = delay
                gBest_feasible_path = path # Store the actual path
                stagnation_counter = 0 # Reset stagnation counter on improvement
                last_best_bn = bn # Update last best known bottleneck
                # print(f"Gen {i+1}: New best feasible BN={bn:.2f}, D={delay:.2f}") # Debug print

        # Update pBests and gBest (based on fitness, not feasibility)
        update_indices = current_fitness > pBests_fitness
        pBests[update_indices] = swarms[update_indices]
        pBests_fitness[update_indices] = current_fitness[update_indices]
        
        current_best_idx = np.argmax(current_fitness)
        if current_fitness[current_best_idx] > gBest_fitness:
            gBest_fitness = current_fitness[current_best_idx]
            gBest = np.copy(swarms[current_best_idx])
        
        # Update velocities and positions
        r1, r2 = np.random.rand(2, num_par, 1)
        velocities = (w * velocities + 
                      c1 * r1 * (pBests - swarms) + 
                      c2 * r2 * (gBest.reshape(1, -1) - swarms))
        swarms += velocities

        # Convergence check (only if a feasible solution has been found)
        if gBest_feasible_path is not None:
            # Check if the best feasible solution has improved this generation
            # Note: gBest_feasible_bn is updated inside the particle loop
            if gBest_feasible_bn <= last_best_bn: # Check if NOT improved
                 stagnation_counter += 1
            # else: # Improvement happened, counter already reset
                 # last_best_bn = gBest_feasible_bn # Update happens inside particle loop too

            if stagnation_counter >= CONVERGENCE_THRESHOLD_GEN:
                terminated_reason = "convergence"
                final_gen = i + 1 # Record generation where convergence occurred
                print(f"INFO: PSO converged after {final_gen} generations (stagnated for {CONVERGENCE_THRESHOLD_GEN}).")
                break
        
        # Ensure last_best_bn reflects the current state for the next iteration's check
        if gBest_feasible_path is not None:
             last_best_bn = gBest_feasible_bn


    pso_time = time.time() - pso_start_time # Final PSO time
    
    print(f"--- [PSO Result] ---")
    if gBest_feasible_path:
        print(f"Execution Time: {pso_time:.4f} sec, Final Gen: {final_gen}, Reason: {terminated_reason}")
        print(f"Found Feasible Path: BN={gBest_feasible_bn:.2f}, D={gBest_feasible_delay:.2f}")
    else:
        # Ensure BN and Delay are -1 if no feasible path was ever found
        gBest_feasible_bn = -1
        gBest_feasible_delay = -1
        print(f"Execution Time: {pso_time:.4f} sec, Final Gen: {final_gen}, Reason: {terminated_reason}")
        print(f"No feasible path found.")
        
    # ★★★ 修正点: 戻り値の数を修正 (8つ) ★★★
    return (opt_bn, lc_time, 
            gBest_feasible_bn, pso_time, gBest_feasible_delay,
            gBest_feasible_path is not None, final_gen, terminated_reason)

if __name__ == '__main__':
    results = []
    num_simulation = 10 
    node_counts = [2600, 2800,3000,3200,3400, 3600,3800,4000] 
    graph_types = ['rnd'] 
    
    constraints = {'delay_multiplier': 3.0} 

    pso_params = {
        'num_par': 200, 'num_gen': 500, # Max generations
        'convergence_gen': 30,         # Convergence threshold
        'time_limit_sec': 1200,         # Timeout in seconds (5 minutes)
        'w_config': (0.9, 0.4),
        'c1_config': (2.5, 0.5),
        'c2_config': (0.5, 2.5),
        'Pd_config': (0.1, 1.0) # Penalty for delay
    }

    # Simulation loop
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

                # ★★★ 修正点: 戻り値のアンパックを修正 (8つ) ★★★
                (opt_bn, opt_time, 
                 pso_bn, pso_time, pso_delay,
                 pso_feasible, pso_final_gen, pso_term_reason
                ) = simulation(Graph, src, dst, constraints, pso_params)

                # Append results, ensuring feasibility check from exact algorithm too
                # Use opt_time (lc_time) correctly
                if opt_bn != -1: # Record only if exact solution could find a feasible path
                     results.append([num_nodes, graph_type, i + 1, 
                                     opt_bn, opt_time, # Use opt_time here
                                     pso_bn, pso_time, pso_delay,
                                     pso_feasible, pso_final_gen, pso_term_reason])

    # --- Write results to CSV ---
    result_path = savef.create_dir(dir_name="2criteria_adaptive_results") 
    filename = result_path + f'/{datetime.now().strftime("%Y%m%d_%H%M%S")}_2criteria_adaptive.csv'
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Num_Nodes', 'Graph_Type', 'Iter', 
                         'Optimal_BN', 'Optimal_Time', # Label-Correcting results
                         'PSO_BN', 'PSO_Time', 'PSO_Delay', # PSO results
                         'PSO_Feasible', 'PSO_Final_Gen', 'PSO_Termination_Reason']) # PSO termination info
        writer.writerows(results)
    print(f"\nSimulation results saved to '{filename}'")

