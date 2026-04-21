# watase_32_optuna.py
# 目的: Optunaを用いてPSOのハイパーパラメータを自動最適化する
# 構成: 空間的近傍(Step1) + Restart機能 + 動的ペナルティ

import optuna
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
import sys

# --- 1. 高速化されたPathEncode (変更なし) ---
def PathEncode(Particle, Graph, src, dst):
    path = [src]
    current_node = src
    visited = {src}
    limit_len = len(Particle)
    
    while current_node != dst:
        # リスト内包表記で高速フィルタリング
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

# --- 2. 属性計算 (変更なし) ---
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

# --- 3. 空間的近傍 (Step 1) ---
def get_spatial_lbest(swarms, pBests, pBests_fitness, k=5):
    num_par = swarms.shape[0]
    sq_norms = np.sum(swarms**2, axis=1)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * np.dot(swarms, swarms.T)
    dist_sq = np.maximum(dist_sq, 0)
    
    # 上位k個の近傍を選択
    nearest_indices = np.argpartition(dist_sq, kth=k, axis=1)[:, :k]
    
    lBest_matrix = np.zeros_like(swarms)
    for i in range(num_par):
        neighbors = nearest_indices[i]
        neighbor_fitness = pBests_fitness[neighbors]
        best_local_idx = np.argmax(neighbor_fitness)
        best_global_idx = neighbors[best_local_idx]
        lBest_matrix[i] = pBests[best_global_idx]
        
    return lBest_matrix

# --- 4. PSOシミュレーション関数 (Restart & 動的ペナルティ対応) ---
def simulation(Graph, src, dst, constraints, pso_params):
    # Optuna探索では厳密解法の計算は不要だが、比較用に計算しておいても良い
    # ここでは高速化のため厳密解法部分はスキップし、PSOの結果のみを返す
    
    max_delay = constraints['delay_multiplier'] # Dijkstraなどは省略
    # ※注意: 正確なmax_delayを得るにはDijkstraが必要ですが、
    # create_graph_4の設定値から概算するか、あるいは毎回計算するか。
    # ここでは正確性を期すためDijkstraを一回走らせます。
    try:
        min_delay_path = nx.dijkstra_path(Graph, source=src, target=dst, weight='delay')
        _, min_delay, _, _ = calculate_path_attributes_4d(Graph, min_delay_path)
        max_delay = min_delay * constraints['delay_multiplier']
    except:
        return -1.0 # パスなし

    max_loss = constraints['loss_constraint']
    min_rel = constraints['reliability_constraint']

    # --- PSO ---
    num_nodes, num_par, num_gen = len(Graph.nodes()), pso_params['num_par'], pso_params['num_gen']
    w_start, w_end = pso_params['w_config']; c1_start, c1_end = pso_params['c1_config']
    c2_start, c2_end = pso_params['c2_config']; Pd_start, Pd_end = pso_params['Pd_config']
    Pl_start, Pl_end = pso_params['Pl_config']; Pr_start, Pr_end = pso_params['Pr_config']
    
    TIME_LIMIT_SEC = pso_params['time_limit_sec']
    pso_start_time = time.time()
    
    # Restart用パラメータ
    RESTART_THRESHOLD = 20
    restart_counter = 0

    swarms = np.random.uniform(1, 20, (num_par, num_nodes)); velocities = np.zeros_like(swarms)
    pBests = np.copy(swarms); pBests_fitness = np.full(num_par, -float('inf'))
    gBest = swarms[0]; gBest_fitness = -float('inf')
    gBest_feasible_bn = -1; gBest_feasible_path = None
    
    last_best_bn = -1.0 

    for i in range(num_gen):
        if time.time() - pso_start_time > TIME_LIMIT_SEC: break

        # --- 停滞検知 & Restart ---
        if gBest_feasible_bn > last_best_bn:
            restart_counter = 0; last_best_bn = gBest_feasible_bn
        else:
            restart_counter += 1
            
        if restart_counter >= RESTART_THRESHOLD:
            # 爆発 (再初期化)
            swarms = np.random.uniform(1, 20, (num_par, num_nodes))
            velocities = np.zeros_like(swarms)
            if gBest is not None: swarms[0] = gBest # エリート保存
            pBests = np.copy(swarms) # 記憶リセット
            pBests_fitness = np.full(num_par, -float('inf'))
            restart_counter = 0
        # ------------------------

        # 動的パラメータ計算
        progress = i / num_gen
        w = w_start - (w_start - w_end) * progress
        c1 = c1_start - (c1_start - c1_end) * progress
        c2 = c2_start + (c2_end - c2_start) * progress
        P_d = Pd_start + (Pd_end - Pd_start) * progress
        P_l = Pl_start + (Pl_end - Pl_start) * progress
        P_r = Pr_start + (Pr_end - Pr_start) * progress
        
        current_fitness = np.zeros(num_par)
        
        for j in range(num_par):
            path, is_valid = PathEncode(swarms[j], Graph, src, dst)
            if not is_valid: 
                current_fitness[j] = -1.0; continue
            
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

        update_indices = current_fitness > pBests_fitness
        pBests[update_indices] = swarms[update_indices]; pBests_fitness[update_indices] = current_fitness[update_indices]
        
        current_best_idx = np.argmax(current_fitness)
        if current_fitness[current_best_idx] > gBest_fitness:
            gBest_fitness = current_fitness[current_best_idx]; gBest = swarms[current_best_idx]
        
        # 空間的近傍
        lBest_matrix = get_spatial_lbest(swarms, pBests, pBests_fitness, k=5)
        
        r1, r2 = np.random.rand(2, num_par, 1)
        velocities = w * velocities + c1 * r1 * (pBests - swarms) + c2 * r2 * (lBest_matrix - swarms)
        swarms += velocities

    return gBest_feasible_bn

# --- 5. Optuna 目的関数 ---
def objective(trial):
    # ==============================
    # 探索空間の定義 (Search Space)
    # ==============================
    
    # 1. ノード数 (Step 1なので 300程度でスクリーニング)
    TARGET_NUM_NODES = 100 
    
    # 2. 粒子数 (計算時間との兼ね合いで300~800)
    FIXED_NUM_PAR = 10
    
    # 3. 速度パラメータ (動的)
    w_start = 0.9
    w_end = trial.suggest_float('w_end', 0.1, 0.6)
    
    c1_start = trial.suggest_float('c1_start', 0.5, 3.5)
    c1_end = trial.suggest_float('c1_end', 0.1, 1.5)
    
    c2_start = trial.suggest_float('c2_start', 0.5, 3.5)
    c2_end = trial.suggest_float('c2_end', 1.5, 4.0)
    
    # 4. ペナルティ (対数スケール log=True)
    # Start: 甘め (1.0 ~ 100.0)
    # End: 厳しめ (1000.0 ~ 100000.0)
    pl_start = trial.suggest_float('pl_start', 1.0, 100.0, log=True)
    pl_end = trial.suggest_float('pl_end', 1000.0, 100000.0, log=True)
    
    # Delayペナルティも最適化対象にする
    pd_start = trial.suggest_float('pd_start', 0.1, 10.0)
    pd_end = trial.suggest_float('pd_end', 10.0, 500.0)

    pso_params = {
        'num_par': FIXED_NUM_PAR, 
        'num_gen': 100, # チューニング時は100世代固定
        'time_limit_sec': 300, 
        'w_config': (w_start, w_end),
        'c1_config': (c1_start, c1_end),
        'c2_config': (c2_start, c2_end),
        'Pd_config': (pd_start, pd_end),
        'Pl_config': (pl_start, pl_end),
        'Pr_config': (pl_start, pl_end) # ReliabilityはLossと同じ設定を使う
    }
    
    constraints = {'delay_multiplier': 3.0, 'loss_constraint': 0.1, 'reliability_constraint': 0.95}

    # ==============================
    # 評価 (3回平均)
    # ==============================
    scores = []
    
    # 3回異なるグラフで実行して、運要素を排除する
    for _ in range(50):
        Graph = create_graph.random_graph(num_node=TARGET_NUM_NODES)
        
        # グラフが連結で十分な大きさかチェック
        if not nx.is_connected(Graph):
             largest_cc = max(nx.connected_components(Graph), key=len)
             node_list = list(largest_cc)
        else:
             node_list = list(Graph.nodes())
             
        if len(node_list) < 2:
            scores.append(0)
            continue
            
        src, dst = random.sample(node_list, 2)
        
        try:
            # シミュレーション実行 (戻り値はBN)
            bn = simulation(Graph, src, dst, constraints, pso_params)
            
            # 解が見つからなかった場合(-1)は、非常に低いスコアを与える
            if bn == -1:
                scores.append(0.0)
            else:
                scores.append(bn)
        except Exception as e:
            # エラー時は0点
            scores.append(0.0)
            
    # 平均スコアを返す
    return sum(scores) / len(scores)


if __name__ == '__main__':
    # 結果保存用ディレクトリ
    result_path = savef.create_dir(dir_name="optuna_results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("Optunaによるパラメータ探索を開始します (N=300)...")
    
    # 最適化の実行 (最大化問題)
    study = optuna.create_study(direction='maximize')
    
    # 試行回数: 50回〜100回推奨 (時間はかかります)
    # ここでは例として50回に設定
    study.optimize(objective, n_trials=100)

    print("\n" + "="*30)
    print("探索終了！ 最強のパラメータ:")
    print("="*30)
    print(study.best_params)
    print(f"最高スコア (平均BN): {study.best_value}")
    
    # CSVに保存
    csv_filename = f"{result_path}/optuna_best_params_{timestamp}.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Parameter', 'Value'])
        for key, value in study.best_params.items():
            writer.writerow([key, value])
        writer.writerow(['Best_Score', study.best_value])
        
    print(f"\n結果を '{csv_filename}' に保存しました。")
    print("次のステップ: このパラメータを使って、N=1000の本番シミュレーションを行ってください。")