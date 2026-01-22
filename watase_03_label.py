import numpy as np
import random
import networkx as nx
import create_graph
from datetime import datetime
import csv
import savef 
from heapq import heappop, heappush

"""
PSO: with comparison (label-correcting algorithm for optimal benchmark)
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
# ★★★ ここからが、厳密解を求める「ラベル訂正アルゴリズム」です ★★★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
def find_optimal_path_by_label_correcting(
    G, source, target, max_delay, weight="weight", delay="delay"
):
    """
    ラベル訂正アルゴリズムを用いて、遅延制約下でボトルネックを最大化する
    厳密な最適経路を探索する。

    Args:
        G (nx.Graph): グラフオブジェクト
        source: 開始ノード
        target: 終了ノード
        max_delay: 最大許容遅延
        weight (str): ボトルネック容量を表すエッジ属性
        delay (str): 遅延を表すエッジ属性

    Returns:
        tuple: (最適経路のリスト, 最大ボトルネック, その時の遅延)。
               経路が見つからない場合は (None, -1, -1)
    """
    # labels[node] = [ (delay_1, bottle_1), (delay_2, bottle_2), ... ]
    # 各ノードへの、互いに支配されない（非劣な）ラベルのリストを保持
    labels = {node: [] for node in G.nodes()}

    # pred[node][label] = (parent_node, parent_label): 経路復元用
    pred = {node: {} for node in G.nodes()}

    # 優先度付きキュー: (-ボトルネック, 遅延, 現在のノード, 現在のラベル)
    pq = []

    # 開始ノードの初期化
    initial_delay = 0.0
    initial_bottle = float('inf')
    start_label = (initial_delay, initial_bottle)
    
    labels[source].append(start_label)
    pred[source][start_label] = (None, None)
    heappush(pq, (-initial_bottle, initial_delay, source, start_label))

    while pq:
        neg_bottle, current_delay, u, current_label = heappop(pq)
        current_bottle = -neg_bottle

        # キューから取り出した経路が、より良い経路が後から見つかったことで
        # 時代遅れ（支配されている）になっている場合はスキップ
        if current_label not in labels[u]:
            continue

        # 隣接ノードを探索
        for v in G.neighbors(u):
            edge_data = G.get_edge_data(u, v)
            edge_weight = edge_data.get(weight, 1)
            edge_delay = edge_data.get(delay, 0)

            new_delay = current_delay + edge_delay
            new_bottle = min(current_bottle, edge_weight)
            new_label = (new_delay, new_bottle)

            # (A) 遅延制約による枝刈り
            if new_delay > max_delay:
                continue

            # (B) 優越関係のチェック: vへの新しい経路が、既存のどの経路にも支配されていないか確認
            is_dominated = False
            for d, b in labels[v]:
                if d <= new_delay and b >= new_bottle:
                    is_dominated = True
                    break
            
            if is_dominated:
                continue

            # --- 新しい非劣経路が見つかったので、リストを更新 ---

            # 1. 既存のラベルから、新しいラベルに支配されるようになったものを削除
            labels[v] = [(d, b) for d, b in labels[v] if not (d >= new_delay and b <= new_bottle)]
            
            # 2. 新しい非劣ラベルを追加
            labels[v].append(new_label)
            
            # 3. 経路復元情報を記録し、キューに追加
            pred[v][new_label] = (u, current_label)
            heappush(pq, (-new_bottle, new_delay, v, new_label))

    # --- 探索終了後、最終的な最適解を選択 ---
    final_labels_at_target = labels.get(target, [])

    if not final_labels_at_target:
        return None, -1, -1

    # 遅延制約を満たす解の中から、ボトルネックが最大のものを探す
    best_bottle = -1
    best_solution_label = None
    for d, b in final_labels_at_target:
        if d <= max_delay:
            if b > best_bottle:
                best_bottle = b
                best_solution_label = (d, b)

    if best_solution_label is None:
        # 制約を満たす経路が一つもなかった
        return None, -1, -1
        
    # --- 最適経路を復元 ---
    path = []
    curr_node = target
    curr_label = best_solution_label
    while curr_node is not None:
        path.append(curr_node)
        parent_node, parent_label = pred[curr_node][curr_label]
        curr_node = parent_node
        curr_label = parent_label
    path.reverse()
    
    final_delay, final_bottle = best_solution_label
    return path, final_bottle, final_delay
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★ ここまでが、厳密解を求める「ラベル訂正アルゴリズム」です ★★★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

# シミュレーションを行う関数
def simulation(Graph, gen, par, delay_multiplier=1.5):
    num_node = int(len(Graph.nodes()))
    num_par = par
    num_gen = gen
    lb, ub = 1, 20
    c_1 = 0.7
    c_2 = 0.3
    
    node_list = list(range(num_node))
    src_node = random.choice(node_list)
    node_list.remove(src_node)
    dst_node = random.choice(node_list)

    print('始点->終点: ' + str(src_node) + '->' + str(dst_node))

    # 相対的な最大許容遅延の計算
    try:
        min_delay = nx.dijkstra_path_length(Graph, source=src_node, target=dst_node, weight='delay')
        print(f"最短遅延 (Delay-based Dijkstra): {min_delay:.2f}")
    except nx.NetworkXNoPath:
        print("エラー: 始点・終点間に経路が存在しません。このシミュレーションはスキップします。")
        return 0, 0, 0, 0, 0

    max_allowable_delay = min_delay * delay_multiplier
    print(f"最大許容遅延 (最短遅延の {delay_multiplier} 倍): {max_allowable_delay:.2f}")

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

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★ 変更点: 正しいラベル訂正アルゴリズムで最適解を計算 ★★★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    label_correcting_bn = 0
    try:
        optimal_path, optimal_bn, optimal_delay = find_optimal_path_by_label_correcting(
            Graph, source=src_node, target=dst_node, max_delay=max_allowable_delay
        )
        if optimal_path is not None:
            label_correcting_bn = optimal_bn
            print('<Label-Correcting Algorithm (The True Optimal)>')
            print(f'BottleNeck: {optimal_bn:.2f}, Delay: {optimal_delay:.2f}, Path: {optimal_path}')
        else:
            print('<Label-Correcting Algorithm (The True Optimal)>')
            print(f'No path found from {src_node} to {dst_node} within delay {max_allowable_delay:.2f}')
    except Exception as e:
        print('<Label-Correcting Algorithm (The True Optimal)>')
        print(f"An error occurred: {e}")


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
    
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★ 変更点: 戻り値の変数を変更 ★★★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    return gBests_bns[l], bn_inv, gBests_delays[l], mod_dijkstra_bn, label_correcting_bn

###############
### main関数 ###
###############
if __name__ == '__main__':
    num_simulation = 50
    num_nodes = 100
    num_par = 500
    num_gen = 1000

    result_path = savef.create_dir()
    savef.make_doc(result_path, num_node=num_nodes, num_par=num_par, num_gen=num_gen)
    
    delay_multipliers = [1.2, 1.5, 2.0]
    results = []

    # 各遅延倍率でループを実行
    for multiplier in delay_multipliers:
        print(f"\n========================================================")
        print(f"===== 遅延倍率: x{multiplier} でのシミュレーションを開始 =====")
        print(f"========================================================")
        
        # 各倍率で、指定された回数のシミュレーションを実行
        for iter_num in range(num_simulation):
            print(f'\n--- 試行回数: {iter_num+1}/{num_simulation} (倍率: x{multiplier}) ---')
            Graph = create_graph.rnd_graph(num_node=num_nodes)
            
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★ 変更点: 戻り値を受け取る変数を変更 ★★★
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            pso_bn, inv_dijkstra_bn, pso_delay, mod_dijkstra_bn, label_correcting_bn = simulation(
                Graph=Graph, gen=num_gen, par=num_par, delay_multiplier=multiplier
            )
            
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★ 変更点: 結果リストに追加する変数を変更 ★★★
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            results.append([
                iter_num + 1, multiplier, pso_bn, inv_dijkstra_bn,
                mod_dijkstra_bn, label_correcting_bn, pso_delay
            ])

    file_name = datetime.now().strftime('%Y%m%d_%H%M%S') + '_with_relative_delay.csv'
    file_path = result_path + '/' + file_name

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★ 変更点: CSVヘッダーの列名を変更 ★★★
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        writer.writerow([
            'iter', 'delay_multiplier', 'PSO_Bottleneck', 'Inverse_Dijkstra_BN',
            'Unconstrained_Mod_Dijkstra_BN', 'Label_Correcting_BN (Optimal)',
            'PSO_Delay'
        ])
        writer.writerows(results)

    print(f"\n結果が '{file_path}' に保存されました。")