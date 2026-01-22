import numpy as np
import random
import networkx as nx
import create_graph 
from datetime import datetime
import csv
import savef # 修正済みのsavef.pyを想定

"""
PSO: without comparison (extended with delay constraint)
"""

###########
### 関数 ###
###########

# ランダムグラフを作成する関数 (create_graph.pyに移動)
# def rnd_graph(n, prob):
#     nodes = list(range(n))
#     edges = []
#     for i in range(n - 1):
#         for j in range(i + 1, n):
#             if random.random() < prob:
#                 edges.append((i, j))
#     return nodes, edges

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
            # 強制的にループを終了させるか、経路を無効と判断する
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
        # エッジが存在し、'weight'属性があるか確認
        if Graph.has_edge(Vk[i], Vk[i+1]) and 'weight' in Graph[Vk[i]][Vk[i+1]]:
            sum_weight += Graph[Vk[i]][Vk[i+1]]['weight']
        else:
            # エッジが存在しないか、重み属性がない場合
            # 通常はPathEncodeでvalid=Falseになるはずだが、念のため非常に大きな値を返すなど
            return float('inf') # 無効な経路として非常に大きな値
    return sum_weight

# ボトルネックリンクを取り出す関数 (最小容量)
def bottleneck(Graph, Vk):
    if len(Vk) < 2: # 経路が短すぎる場合は0または非常に小さい値
        return 0
    
    bn = float('inf') # 初期値を無限大に設定
    for i in range(len(Vk)-1):
        # エッジが存在し、'weight'属性があるか確認
        if Graph.has_edge(Vk[i], Vk[i+1]) and 'weight' in Graph[Vk[i]][Vk[i+1]]:
            temp = Graph[Vk[i]][Vk[i+1]]['weight']
            if temp < bn:
                bn = temp
        else:
            # エッジが存在しないか、重み属性がない場合はボトルネックを0と見なす
            return 0
    
    if bn == float('inf'): # 経路はあったが有効なエッジがなかった場合など
        return 0
    
    return bn

# ★追加★ 経路から合計遅延を計算する関数
def path2delay(Graph, Vk):
    total_delay = 0
    if len(Vk) < 2: # 経路が短すぎる場合は0
        return 0
    for i in range(len(Vk) - 1):
        # エッジが存在し、'delay'属性があるか確認
        if Graph.has_edge(Vk[i], Vk[i+1]) and 'delay' in Graph[Vk[i]][Vk[i+1]]:
            total_delay += Graph[Vk[i]][Vk[i+1]]['delay']
        else:
            # エッジが存在しないか、遅延属性がない場合は、非常に大きな遅延（経路が無効）と見なす
            return float('inf') # 無効な経路として非常に大きな遅延
    return total_delay

# シミュレーションを行う関数
def simulation(Graph, gen, par):
    # 定数
    num_node = int(len(Graph.nodes()))   # ノード数
    num_par = par   # 粒子数
    num_gen = gen   # 世代数
    lb, ub = 1, 20  # 乱数の加減と上限 (優先度リストの値の範囲)
    c_1 = 0.7       # PSOのパラメータ1: pBestの重み
    c_2 = 0.3       # PSOのパラメータ2: gBestの重み
    
    max_allowable_delay = 50.0 # ★追加★: 許容される最大遅延を設定 (例: 50.0ミリ秒)

    # 始点と終点の選択
    node_list = [i for i in range(0, num_node)]     # ノードのリストを生成
    src_node = random.choice(node_list)             # 始点の選択
    node_list.remove(src_node)                      # 始点をリストから除外
    dst_node = random.choice(node_list)             # 終点の選択
    
    print('始点->終点: ' + str(src_node) + '->' + str(dst_node))

    # inverse Dijkstra (比較用)
    # 容量の逆数に対してDijkstraを実行し、ボトルネックリンクを評価
    # このDijkstraは遅延制約を考慮しないため、PSOとの比較対象として残す
    # ただし、'delay'属性が必須となる場合は、Dijkstraの重みに影響する可能性がある
    for (i,j) in Graph.edges():
        if 'weight' in Graph[i][j]:
            temp_inv = 1/Graph[i][j]["weight"] if Graph[i][j]["weight"] != 0 else float('inf')
            Graph[i][j]['inv'] = temp_inv
        else:
            Graph[i][j]['inv'] = float('inf') # weightがない場合は無限大

    try:
        v_inv = nx.dijkstra_path(Graph, source=src_node, target=dst_node, weight='inv')
        bn_inv = bottleneck(Graph, v_inv) # Inverse Dijkstraで見つけたパスのボトルネック容量
        print('<Modified Dijkstra>')
        print('BottleNeck: ', bn_inv, 'path', v_inv)
    except nx.NetworkXNoPath:
        v_inv = []
        bn_inv = 0
        print('<Modified Dijkstra>')
        print('No path found by Modified Dijkstra (inverse capacity).')

    # 変数
    # 粒子群
    swarms = np.zeros((num_gen, num_par, num_node))
    # 速度
    velocities = np.zeros_like(swarms)
    # 経路
    paths = [0] * num_gen
    # 経路妥当フラグ
    valids = [0] * num_gen
    # 重みの合計 (ここでは容量の合計)
    sums = np.zeros((num_gen, num_par))
    # ボトルネック容量
    bns = np.zeros((num_gen, num_par))
    # ★追加★ 遅延の合計
    delays = np.zeros((num_gen, num_par))
    
    # 評価値
    fitness = np.zeros_like(sums)
    # pBestの評価値
    pBests_fitness = np.zeros_like(sums)
    # pBest
    pBests = np.zeros_like(swarms)
    # gBestの評価値
    gBests_fitness = np.zeros(num_gen)
    # gBest
    gBests = np.zeros((num_gen, num_node))
    # gBestの経路
    gBests_paths = [0] * num_gen
    # gBestの経路の重み (容量の合計)
    gBests_sums = [0] * num_gen
    # gBestの経路のボトルネックリンク容量
    gBests_bns = [0] * num_gen
    # ★追加★ gBestの経路の合計遅延
    gBests_delays = [0] * num_gen

    # 初期化
    # 初期粒子の生成[j][k]
    par_rand = np.array([[random.randint(lb, ub) for _ in range(num_node)] for _ in range(num_par)])
    # 初期粒子の格納[0][j][k]
    swarms[0] = par_rand

    # 一時的に格納
    path_list = []
    valid_list = []
    bns_list = []   # ボトルネック容量を格納
    sums_list = []  # リンク容量の合計を格納
    delay_list = [] # ★追加★: 遅延を格納

    for j in range(num_par):
        temp_path, temp_valid = PathEncode(swarms[0][j], Graph, src=src_node, dst=dst_node)
        path_list.append(temp_path)
        valid_list.append(temp_valid)
        
        bns_temp = bottleneck(Graph, temp_path)
        sums_temp = path2weight(Graph, temp_path)
        delay_temp = path2delay(Graph, temp_path) # ★追加★: 遅延を計算

        bns_list.append(bns_temp)
        sums_list.append(sums_temp)
        delay_list.append(delay_temp) # ★追加★

    # 0世代のpaths, valids, bns, sums, delays に値を格納
    paths[0] = path_list
    valids[0] = valid_list
    bns[0] = bns_list
    sums[0] = sums_list
    delays[0] = delay_list # ★追加★

    # 評価値の格納 (ボトルネック容量と遅延制約を考慮)
    fitness[0] = bns[0] # 初期はボトルネック容量
    for j in range(num_par): # num_par を使う方が安全
        if valids[0][j] == False:
            fitness[0][j] = 0 # 無効な経路は適応度0
        # ★追加★: 遅延制約を適用
        elif delays[0][j] > max_allowable_delay:
            fitness[0][j] = 0 # 遅延が制約を超えた場合、適応度を0にする (ペナルティ)

    # pBestの初期化
    for j in range(num_par):
        pBests_fitness[0][j] = fitness[0][j]
        pBests[0][j] = swarms[0][j]
    
    # gBestの初期化
    temp_index = np.argmax(fitness[0]) # 最高の適応度を持つ粒子のインデックス
    temp_value = np.max(fitness[0])    # 最高の適応度値
    
    gBests_fitness[0] = temp_value
    gBests[0] = swarms[0][temp_index]
    gBests_paths[0] = paths[0][temp_index]
    gBests_bns[0] = bns[0][temp_index]
    gBests_sums[0] = sums[0][temp_index]
    gBests_delays[0] = delays[0][temp_index] # ★追加★

    # 最適化ループ
    for i in range(1, num_gen, 1):
        # 速度の更新
        r_1 = random.random()
        r_2 = random.random()
        velocities[i] = velocities[i-1] + c_1 * r_1 * (pBests[i-1] - swarms[i-1]) + c_2 * r_2 * (gBests[i-1] * np.ones_like(swarms[i-1]) - swarms[i-1])
        
        # 位置の更新: 値が乱数の範囲内に収まるようにクリップ
        # intへの変換は、PathEncodeが優先度リストを整数として扱うため必要
        swarms[i] = np.round(swarms[i-1] + velocities[i]).astype(int)
        swarms[i] = np.clip(swarms[i], lb, ub) # lb, ubの範囲にクリップ

        # tempリストをクリア
        path_list = []
        valid_list = []
        bns_list = []
        sum_list = []
        delay_list = [] # ★追加★

        # paths, valids, bns, sums, delays 更新
        for j in range(num_par):
            temp_path, temp_valid = PathEncode(swarms[i][j], Graph, src=src_node, dst=dst_node)
            path_list.append(temp_path)
            valid_list.append(temp_valid)

            bns_temp = bottleneck(Graph, temp_path)
            sum_temp = path2weight(Graph, temp_path)
            delay_temp = path2delay(Graph, temp_path) # ★追加★

            bns_list.append(bns_temp)
            sum_list.append(sum_temp)
            delay_list.append(delay_temp) # ★追加★

        paths[i] = path_list
        valids[i] = valid_list
        bns[i] = bns_list
        sums[i] = sum_list
        delays[i] = delay_list # ★追加★

        # 評価値 (ボトルネック容量と遅延制約を考慮)
        fitness[i] = bns[i] # 初期はボトルネック容量
        for j in range(num_par): # num_par を使う方が安全
            if valids[i][j] == 0:
                fitness[i][j] = 0 # 無効な経路は適応度0
            # ★追加★: 遅延制約を適用
            elif delays[i][j] > max_allowable_delay:
                fitness[i][j] = 0 # 遅延が制約を超えた場合、適応度を0にする

        # pBest 更新
        for j in range(num_par):
            if pBests_fitness[i-1][j] < fitness[i][j]: # 現在の適応度がpBestより良ければ更新
                pBests_fitness[i][j] = fitness[i][j]
                pBests[i][j] = swarms[i][j]
            else: # 悪ければpBestを維持
                pBests_fitness[i][j] = pBests_fitness[i-1][j]
                pBests[i][j] = pBests[i-1][j]
                
        # gBest 更新
        temp_index = np.argmax(fitness[i])  # 評価値が最高の粒子の番号
        temp_value = np.max(fitness[i])     # 最高の評価値
        
        if temp_value > gBests_fitness[i-1]: # 現在世代の最高の適応度が前世代のgBestより良ければ更新
            gBests_fitness[i] = temp_value
            gBests[i] = swarms[i][temp_index]
            gBests_paths[i] = paths[i][temp_index]
            gBests_bns[i] = bns[i][temp_index]
            gBests_sums[i] = sums[i][temp_index]
            gBests_delays[i] = delays[i][temp_index] # ★追加★
        else: # 悪ければgBestを維持
            gBests_fitness[i] = gBests_fitness[i-1]
            gBests[i] = gBests[i-1]
            gBests_paths[i] = gBests_paths[i-1]
            gBests_bns[i] = gBests_bns[i-1]
            gBests_sums[i] = gBests_sums[i-1]
            gBests_delays[i] = gBests_delays[i-1] # ★追加★

    # 結果
    l = num_gen - 1
    print('<PSO>')
    print('BottleNeck: ', gBests_bns[l], 'path', gBests_paths[l], 'delay:', gBests_delays[l]) # ★修正★

    # シミュレーション関数の出力
    # ★修正★: 遅延も返すようにする
    return gBests_bns[num_gen - 1], bn_inv, gBests_delays[num_gen - 1]

###############
### main関数 ###
###############
if __name__ == '__main__':
    num_simulation = 3  # シミュレーション回数
    num_nodes = 300   # ノード数
    num_par = 500    # 粒子数
    num_gen = 1000    # 世代数
    
    # 出力
    result_path = savef.create_dir()    # 結果を格納するディレクトリのパス
    # シミュレーションの説明
    savef.make_doc(result_path, num_node=num_nodes, num_par=num_par, num_gen=num_gen)
    
    bottleneck_results = []   # シミュレーション結果を格納するリスト

    # num_simulation回シミュレーションを行う
    for iter in range(num_simulation):
        print('試行回数:', iter)
        # 毎回グラフを生成
        Graph = create_graph.rnd_graph(num_node=num_nodes) # create_graph.py の rnd_graph を使用
        
        # シミュレーション結果を一時的に出力
        # ★修正★: simulation関数の戻り値に遅延を追加
        pso_bn, dijkstra_bn_inv, pso_delay = simulation(Graph=Graph, gen=num_gen, par=num_par)
        
        # シミュレーション結果を格納
        bottleneck_results.append([iter, pso_bn, dijkstra_bn_inv, pso_delay]) # ★修正★: 遅延も格納
    
    # ファイル名を生成
    file_name = datetime.now().strftime('%Y%m%d_%H%M%S') + '_rich_with_delay.csv'
    file_path = result_path + '/' + file_name
    
    # ファイルに書き出し
    with open(file_path, 'w', newline='') as f: # newline='' を追加すると空白行を避けられる
        writer = csv.writer(f)
        # ★修正★: ヘッダーに遅延を追加
        writer.writerow(['iter', 'PSO_Bottleneck', 'Dijkstra_Bottleneck_Inv', 'PSO_Delay'])
        writer.writerows(bottleneck_results)

    print(f"結果が '{file_path}' に保存されました。")

         