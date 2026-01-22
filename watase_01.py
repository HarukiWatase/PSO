import numpy as np
import random
import networkx as nx
import create_graph
from datetime import datetime
import csv
import savef # 修正済みのsavef.pyを想定
from heapq import heappop, heappush # max_load_pathで必要

"""
PSO: with comparison (extended with delay constraint and its optimal benchmark)
"""

###########
### 関数 ###
###########

# 経路エンコーディング(numpy)
def PathEncode(Particle, Graph, src, dst):
    # 宣言部
    N_inf = -100 # 優先度リストで通過済みノードを示す値
    k = 0
    tk = src # 現在のノード
    Vp = [tk] # 経路リスト
    temp = np.copy(Particle) # 粒子のコピー (優先度リスト)
    temp[tk] = N_inf # 始点を通過済みとしてマーク

    num_node = len(temp)

    # 実行部
    for _ in range(num_node): # ノード数回までループ
        # 終了判定
        if (tk == dst or k >= num_node - 1): # 目的地に到達したか、最大ステップ数を超えたら終了
            break

        # ソート: 優先度が高い順にノードのインデックスを取得
        temp_order = temp.argsort()[::-1]

        # 隣接点の取得
        adjs = list(Graph.adj[tk])

        # すべてのノードについて優先度順に確認
        found_next = False
        for i in range(num_node):
            next_node_candidate = temp_order[i]
            # 隣接点の判定
            if next_node_candidate in adjs:
                # 通過済みの判定
                if temp[next_node_candidate] == N_inf:
                    # このノードは既に通過済み、次の優先度が高いノードを探す
                    continue
                else:
                    # 未通過の隣接点が見つかった
                    tk = next_node_candidate
                    Vp.append(tk)
                    temp[tk] = N_inf # 通過済みとしてマーク
                    k += 1
                    found_next = True
                    break # 次のステップへ

        if not found_next and tk != dst: # どこにも進めなくなったが目的地に到達していない
            break

    # 経路の妥当性
    valid = False
    if tk == dst: # 最終的に目的地に到達していれば有効
        valid = True
    return Vp, valid

# 経路から重みを計算する関数 (リンク容量の合計)
def path2weight(Graph, Vk):
    sum_weight = 0
    if len(Vk) < 2: # 経路が短すぎる場合は0
        return 0
    for i in range(len(Vk) - 1):
        if Graph.has_edge(Vk[i], Vk[i+1]) and 'weight' in Graph[Vk[i]][Vk[i+1]]:
            sum_weight += Graph[Vk[i]][Vk[i+1]]['weight']
        else:
            return float('inf') 
    return sum_weight

# ボトルネックリンクを取り出す関数 (最小容量)
def bottleneck(Graph, Vk):
    if len(Vk) < 2:
        return 0
    bn = float('inf')
    for i in range(len(Vk)-1):
        if Graph.has_edge(Vk[i], Vk[i+1]) and 'weight' in Graph[Vk[i]][Vk[i+1]]:
            temp = Graph[Vk[i]][Vk[i+1]]['weight']
            if temp < bn:
                bn = temp
        else:
            return 0
    if bn == float('inf'):
        return 0
    return bn

# 経路から合計遅延を計算する関数
def path2delay(Graph, Vk):
    total_delay = 0
    if len(Vk) < 2:
        return 0
    for i in range(len(Vk) - 1):
        if Graph.has_edge(Vk[i], Vk[i+1]) and 'delay' in Graph[Vk[i]][Vk[i+1]]:
            total_delay += Graph[Vk[i]][Vk[i+1]]['delay']
        else:
            return float('inf')
    return total_delay

# --- 修正ダイクストラ (制約なし) ---
def max_load_path(G, source, target, weight="weight"):
    if source not in G:
        raise nx.NodeNotFound(f"Source {source} not in graph")
    if target not in G:
        raise nx.NodeNotFound(f"Target {target} not in graph")
    pq = []
    bottleneck_vals = {node: float("-inf") for node in G}
    bottleneck_vals[source] = float("inf")
    pred = {node: None for node in G}
    visited = set()
    heappush(pq, (-bottleneck_vals[source], source))
    while pq:
        curr_bottle_neg, u = heappop(pq)
        curr_bottle = -curr_bottle_neg
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            break
        for v in G.neighbors(u):
            edge_data = G[u][v]
            w = edge_data.get(weight, 1)
            new_bottle = min(curr_bottle, w)
            if new_bottle > bottleneck_vals[v]:
                bottleneck_vals[v] = new_bottle
                pred[v] = u
                heappush(pq, (-new_bottle, v))
    if bottleneck_vals[target] == float("-inf"):
        raise nx.NetworkXNoPath(f"No path found from {source} to {target}")
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = pred[current]
    path.reverse()
    return path

def bottleneck_capacity(graph: nx.Graph, path: list) -> float:
    if not path or len(path) < 2:
        return 0
    min_capacity = float('inf')
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if graph.has_edge(u, v) and 'weight' in graph[u][v]:
            min_capacity = min(min_capacity, graph[u][v]['weight'])
        else:
            return 0
    return min_capacity

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★ 追加: 遅延制約付き修正ダイクストラ法 ★★★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
def max_load_path_with_delay_constraint(
    G, source, target, max_delay, weight="weight", delay="delay"
):
    """
    遅延制約下で、ボトルネック帯域を最大化する絶対的な最適経路を探索する。
    PSOの結果を評価するための「理論上の最適解」を求めるために使用する。
    """
    if source not in G or target not in G:
        raise nx.NodeNotFound("Source or target node not in graph.")
    dist = {node: (float("-inf"), float("inf")) for node in G}
    dist[source] = (float("inf"), 0)
    pq = [(-float("inf"), 0, source, [source])]
    best_path = []
    best_bottleneck = float("-inf")
    while pq:
        neg_bottle, current_delay, u, path = heappop(pq)
        current_bottle = -neg_bottle
        if current_bottle < dist[u][0] and current_delay > dist[u][1]:
            continue
        if u == target:
            if current_bottle > best_bottleneck:
                best_bottleneck = current_bottle
                best_path = path
            continue
        for v in G.neighbors(u):
            edge_data = G[u][v]
            edge_weight = edge_data.get(weight, 1)
            edge_delay = edge_data.get(delay, 0)
            new_delay = current_delay + edge_delay
            if new_delay > max_delay:
                continue
            new_bottle = min(current_bottle, edge_weight)
            if new_bottle > dist[v][0] or (new_bottle == dist[v][0] and new_delay < dist[v][1]):
                dist[v] = (new_bottle, new_delay)
                new_path = path + [v]
                heappush(pq, (-new_bottle, new_delay, v, new_path))
    if not best_path:
        raise nx.NetworkXNoPath(f"No path found from {source} to {target} within delay {max_delay}")
    final_delay = path2delay(G, best_path)
    return best_path, best_bottleneck, final_delay


# シミュレーションを行う関数
def simulation(Graph, gen, par):
    num_node = int(len(Graph.nodes()))
    num_par = par
    num_gen = gen
    lb, ub = 1, 20
    c_1 = 0.7
    c_2 = 0.3
    max_allowable_delay = 20.0

    node_list = list(range(num_node))
    src_node = random.choice(node_list)
    node_list.remove(src_node)
    dst_node = random.choice(node_list)

    print('始点->終点: ' + str(src_node) + '->' + str(dst_node))

    # inverse Dijkstra (比較用)
    for (i,j) in Graph.edges():
        if 'weight' in Graph[i][j]:
            temp_inv = 1/Graph[i][j]["weight"] if Graph[i][j]["weight"] != 0 else float('inf')
            Graph[i][j]['inv'] = temp_inv
        else:
            Graph[i][j]['inv'] = float('inf')
    bn_inv = 0
    try:
        v_inv = nx.dijkstra_path(Graph, source=src_node, target=dst_node, weight='inv')
        bn_inv = bottleneck(Graph, v_inv)
        print('<Inverse Dijkstra>')
        print('BottleNeck: ', bn_inv, 'path', v_inv)
    except nx.NetworkXNoPath:
        v_inv = []
        print('<Inverse Dijkstra>')
        print('No path found by Inverse Dijkstra.')

    # modified Dijkstra (制約なし比較用)
    mod_dijkstra_bn = 0
    try:
        v_mod_dijkstra = max_load_path(Graph, source=src_node, target=dst_node, weight='weight')
        mod_dijkstra_bn = bottleneck_capacity(Graph, v_mod_dijkstra)
        print('<Unconstrained Modified Dijkstra>')
        print('BottleNeck: ', mod_dijkstra_bn, 'path', v_mod_dijkstra)
    except nx.NetworkXNoPath:
        v_mod_dijkstra = []
        print('<Unconstrained Modified Dijkstra>')
        print('No path found by Unconstrained Modified Dijkstra.')

    # ★★★ 追加: 遅延制約付き修正ダイクストラ (絶対的な最適解の計算) ★★★
    constrained_mod_dijkstra_bn = 0
    try:
        v_const, bn_const, d_const = max_load_path_with_delay_constraint(
            Graph, source=src_node, target=dst_node, max_delay=max_allowable_delay
        )
        constrained_mod_dijkstra_bn = bn_const
        print('<Constrained Modified Dijkstra (The True Optimal)>')
        print(f'BottleNeck: {bn_const}, Delay: {d_const}, Path: {v_const}')
    except nx.NetworkXNoPath as e:
        v_const = []
        print('<Constrained Modified Dijkstra (The True Optimal)>')
        print(e)

    # PSO変数初期化
    swarms = np.zeros((num_gen, num_par, num_node))
    velocities = np.zeros_like(swarms)
    paths = [0] * num_gen
    valids = [0] * num_gen
    sums = np.zeros((num_gen, num_par))
    bns = np.zeros((num_gen, num_par))
    delays = np.zeros((num_gen, num_par))
    fitness = np.zeros_like(sums)
    pBests_fitness = np.zeros_like(sums)
    pBests = np.zeros_like(swarms)
    gBests_fitness = np.zeros(num_gen)
    gBests = np.zeros((num_gen, num_node))
    gBests_paths = [0] * num_gen
    gBests_sums = [0] * num_gen
    gBests_bns = [0] * num_gen
    gBests_delays = [0] * num_gen

    # PSO初期化
    swarms[0] = np.array([[random.randint(lb, ub) for _ in range(num_node)] for _ in range(num_par)])
    
    path_list, valid_list, bns_list, sums_list, delay_list = [], [], [], [], []
    for j in range(num_par):
        temp_path, temp_valid = PathEncode(swarms[0][j], Graph, src=src_node, dst=dst_node)
        path_list.append(temp_path)
        valid_list.append(temp_valid)
        bns_list.append(bottleneck(Graph, temp_path))
        sums_list.append(path2weight(Graph, temp_path))
        delay_list.append(path2delay(Graph, temp_path))
    paths[0], valids[0], bns[0], sums[0], delays[0] = path_list, valid_list, bns_list, sums_list, delay_list

    fitness[0] = bns[0]
    for j in range(num_par):
        if not valids[0][j] or delays[0][j] > max_allowable_delay:
            fitness[0][j] = 0
    
    for j in range(num_par):
        pBests_fitness[0][j] = fitness[0][j]
        pBests[0][j] = swarms[0][j]

    temp_index = np.argmax(fitness[0])
    gBests_fitness[0] = np.max(fitness[0])
    gBests[0] = swarms[0][temp_index]
    gBests_paths[0] = paths[0][temp_index]
    gBests_bns[0] = bns[0][temp_index]
    gBests_sums[0] = sums[0][temp_index]
    gBests_delays[0] = delays[0][temp_index]

    # 最適化ループ
    for i in range(1, num_gen):
        r_1, r_2 = random.random(), random.random()
        velocities[i] = velocities[i-1] + c_1 * r_1 * (pBests[i-1] - swarms[i-1]) + c_2 * r_2 * (gBests[i-1] - swarms[i-1])
        swarms[i] = np.round(swarms[i-1] + velocities[i]).astype(int)
        swarms[i] = np.clip(swarms[i], lb, ub)
        
        path_list, valid_list, bns_list, sum_list, delay_list = [], [], [], [], []
        for j in range(num_par):
            temp_path, temp_valid = PathEncode(swarms[i][j], Graph, src=src_node, dst=dst_node)
            path_list.append(temp_path)
            valid_list.append(temp_valid)
            bns_list.append(bottleneck(Graph, temp_path))
            sum_list.append(path2weight(Graph, temp_path))
            delay_list.append(path2delay(Graph, temp_path))
        paths[i], valids[i], bns[i], sums[i], delays[i] = path_list, valid_list, bns_list, sum_list, delay_list

        fitness[i] = bns[i]
        for j in range(num_par):
            if not valids[i][j] or delays[i][j] > max_allowable_delay:
                fitness[i][j] = 0

        for j in range(num_par):
            if pBests_fitness[i-1][j] < fitness[i][j]:
                pBests_fitness[i][j] = fitness[i][j]
                pBests[i][j] = swarms[i][j]
            else:
                pBests_fitness[i][j] = pBests_fitness[i-1][j]
                pBests[i][j] = pBests[i-1][j]

        temp_index = np.argmax(fitness[i])
        if np.max(fitness[i]) > gBests_fitness[i-1]:
            gBests_fitness[i] = np.max(fitness[i])
            gBests[i] = swarms[i][temp_index]
            gBests_paths[i] = paths[i][temp_index]
            gBests_bns[i] = bns[i][temp_index]
            gBests_sums[i] = sums[i][temp_index]
            gBests_delays[i] = delays[i][temp_index]
        else:
            gBests_fitness[i] = gBests_fitness[i-1]
            gBests[i] = gBests[i-1]
            gBests_paths[i] = gBests_paths[i-1]
            gBests_bns[i] = gBests_bns[i-1]
            gBests_sums[i] = gBests_sums[i-1]
            gBests_delays[i] = gBests_delays[i-1]

    l = num_gen - 1
    print('<PSO>')
    print('BottleNeck: ', gBests_bns[l], 'path', gBests_paths[l], 'delay:', gBests_delays[l])

    # ★修正★ シミュレーション関数の戻り値を更新
    return gBests_bns[l], bn_inv, gBests_delays[l], mod_dijkstra_bn, constrained_mod_dijkstra_bn

###############
### main関数 ###
###############
if __name__ == '__main__':
    num_simulation = 3
    num_nodes = 100
    num_par = 500
    num_gen = 1000

    result_path = savef.create_dir()
    savef.make_doc(result_path, num_node=num_nodes, num_par=num_par, num_gen=num_gen)

    results = []

    for iter in range(num_simulation):
        print(f'\n--- 試行回数: {iter+1}/{num_simulation} ---')
        Graph = create_graph.rnd_graph(num_node=num_nodes)

        # ★修正★ simulation関数の戻り値(5つ)を正しく受け取る
        pso_bn, inv_dijkstra_bn, pso_delay, mod_dijkstra_bn, constrained_mod_dijkstra_bn = simulation(Graph=Graph, gen=num_gen, par=num_par)
        
        # ★修正★ 結果リストに制約付き最適解も追加
        results.append([iter + 1, pso_bn, inv_dijkstra_bn, mod_dijkstra_bn, constrained_mod_dijkstra_bn, pso_delay])

    # ★修正★ ファイル名をより分かりやすく変更
    file_name = datetime.now().strftime('%Y%m%d_%H%M%S') + '_with_constrained_optimal.csv'
    file_path = result_path + '/' + file_name

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # ★修正★ CSVヘッダーを更新
        writer.writerow([
            'iter', 'PSO_Bottleneck', 'Inverse_Dijkstra_BN', 
            'Unconstrained_Mod_Dijkstra_BN', 'Constrained_Mod_Dijkstra_BN (Optimal)', 
            'PSO_Delay'
        ])
        writer.writerows(results)

    print(f"\n結果が '{file_path}' に保存されました。")