import numpy as np
import random
import networkx as nx
import create_graph
from datetime import datetime
import csv
import savef

"""
PSO: without comparison
"""

###########
### 関数 ###
###########

# ランダムグラフを作成する関数
def rnd_graph(n, prob):
    nodes = list(range(n))
    edges = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if random.random() < prob:
                edges.append((i, j))
    return nodes, edges

# 経路エンコーディング(numpy)
def PathEncode(Particle, Graph, src, dst):
    # 宣言部
    N_inf = -100
    k = 0
    tk = src
    Vp = [tk]
    temp = np.copy(Particle)
    temp[tk] = N_inf
    # temp[0] = N_inf # 優先度リストの0を初期化か
    num_node = len(temp)
    # 実行部
    for _ in range(num_node):
        # 終了判定
        if (tk == dst or k > num_node - 1):
            break
        # ソート
        temp_order = temp.argsort()[::-1]
        # 隣接点の取得
        adjs = list(Graph.adj[tk])
        # すべてのノードについて
        for i in range(num_node):
            # 隣接点の判定
            if temp_order[i] in adjs:
                # 通過済みの判定
                if temp[temp_order[i]] == N_inf:
                    break
                else:
                    tk = temp_order[i]
                    Vp.append(tk)
                    temp[tk] = N_inf
                    k += 1
                    break
    # 経路の妥当性
    valid = False
    if tk == dst:
        valid = True
    return Vp, valid

# 経路から重みを計算する関数
def path2weight(Graph, Vk):
    sum = 0
    for i in range(len(Vk) - 1):
        sum += Graph[Vk[i]][Vk[i+1]]['weight']
        
    return sum

# ボトルネックリンクを取り出す関数
def bottleneck(Graph, Vk):
    bn = 100
    for i in range(len(Vk)-1):
        temp = Graph[Vk[i]][Vk[i+1]]['weight']
        if temp < bn:
            bn = temp
    
    return bn

# シミュレーションを行う関数
def simulation(Graph, gen, par):
    # 定数
    num_node = int(len(Graph.nodes()))   # ノード数
    num_par = par   # 粒子数
    num_gen = gen   # 世代数
    lb, ub = 1, 20  # 乱数の加減と上限
    c_1 = 0.7       # PSOのパラメータ1: pBest
    c_2 = 0.3       # PSOのパラメータ2: gBest
    # 始点と終点の選択
    node_list = [i for i in range(0, num_node)]     # ノードのリストを生成
    src_node = random.choice(node_list)             # 始点の選択
    node_list.remove(src_node)                      # 始点をリストから除外
    dst_node = random.choice(node_list)             # 終点の選択
    # 恣意的な最適経路を追加
    # create_graph.add_optimal_path(Graph=Graph, src=src_node, dst=dst_node)
    # print('始点->終点: ' + str(src_node) + '->' + str(dst_node))
    # inverse Dijkstra
    for (i,j) in Graph.edges():
        temp = 1/Graph[i][j]["weight"]
        Graph[i][j]['inv'] = temp
    # 容量の逆数に対するDijkstraの実行
    v_inv = nx.dijkstra_path(Graph, source=src_node, target=dst_node, weight='inv')
    bn_inv = bottleneck(Graph, v_inv)
    print('<Modified Dijkstra>')
    print('BottleNeck: ', bn_inv, 'path', v_inv)
    
    # 変数
    # 粒子群
    swarms = np.zeros((num_gen, num_par, num_node))
    # 速度
    velocities = np.zeros_like(swarms)
    # 経路
    paths = [0] * num_gen
    # 経路妥当フラグ
    valids = [0] * num_gen
    # 重みの合計
    sums = np.zeros((num_gen, num_par))
    # ボトルネック容量
    bns = np.zeros((num_gen, num_par))
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
    # gBestの経路の重み
    gBests_sums = [0] * num_gen
    # gBestの経路のボトルネックリンク容量
    gBests_bns = np.zeros(num_gen) # ★変更：[0] * num_gen から np.zeros(num_gen) へ（NumPy配列で統一）

    # 初期化
    # 初期粒子の生成[j][k]
    par_rand = np.array([[random.randint(lb, ub) for _ in range(num_node)] for _ in range(num_par)])
    # par_rand = np.array([[random.uniform(lb, ub) for _ in range(num_node)] for _ in range(num_par)])
    # 初期粒子の格納[0][j][k]
    swarms[0] = par_rand

    # 一時的に格納
    path_list = []
    valid_list = []
    for j in range(num_par):
        temp_path, temp_valid = PathEncode(swarms[0][j], Graph, src=src_node, dst=dst_node)
        path_list.append(temp_path)
        valid_list.append(temp_valid)
    # 0世代のpaths, validsに値を格納
    paths[0] = path_list
    valids[0] = valid_list
    
    # ボトルネックリンク問題
    bns_list = []   # ボトルネック容量を格納
    sums_list = []  # リンク容量の合計を格納
    for j in range(len(swarms[0])):
        bns_temp = bottleneck(Graph, paths[0][j])
        sums_temp = path2weight(Graph, paths[0][j])
        bns_list.append(bns_temp)
        sums_list.append(sums_temp)
    bns[0] = bns_list
    sums[0] = sums_list
    # 評価値の格納
    fitness[0] = bns[0]
    for j in range(len(swarms[0])):
        if valids[0][j] == False:
            fitness[0][j] = 0

    # pBest
    for j in range(num_par):
        pBests_fitness[0][j] = fitness[0][j]
        pBests[0][j] = swarms[0][j]
    # gBest
    temp_index = np.argmax(fitness[0])
    temp_value = np.max(fitness[0])
    # i > 0 では比較(temp_value)
    gBests_fitness[0] = temp_value
    gBests[0] = swarms[0][temp_index]
    gBests_paths[0] = paths[0][temp_index]
    gBests_bns[0] = bns[0][temp_index]
    gBests_sums[0] = sums[0][temp_index]
    
    # 最適化
    for i in range(1, num_gen, 1):
        # 速度の更新
        r_1 = random.random()
        r_2 = random.random()
        velocities[i] = velocities[i-1] + c_1 * r_1 * (pBests[i-1] - swarms[i-1]) + c_2 * r_2 * (gBests[i-1] * np.ones_like(swarms[i-1]) - swarms[i-1])
        # 位置の更新
        swarms[i] = swarms[i-1] + velocities[i]

        # temp
        path_list = []
        valid_list = []
        bns_list = []
        sum_list = []

        # paths, valids 更新
        for j in range(num_par):
            temp_path, temp_valid = PathEncode(swarms[i][j], Graph, src=src_node, dst=dst_node)
            path_list.append(temp_path)
            valid_list.append(temp_valid)
        paths[i] = path_list
        valids[i] = valid_list

        # bns, sums, fitness 更新
        for j in range(len(swarms[i])):
            bns_temp = bottleneck(Graph, paths[i][j])
            sum_temp = path2weight(Graph, paths[i][j])
            bns_list.append(bns_temp)
            sum_list.append(sum_temp)
        bns[i] = bns_list
        sums[i] = sum_list
        # 評価値
        fitness[i] = bns[i]
        for j in range(len(swarms[i])):
            if valids[i][j] == 0:
                fitness[i][j] = 0

        # pBest 更新
        for j in range(num_par):
            if pBests_fitness[i-1][j] < fitness[i][j]:
                pBests_fitness[i][j] = fitness[i][j]
                pBests[i][j] = swarms[i][j]
            else:
                pBests_fitness[i][j] = pBests_fitness[i-1][j]
                pBests[i][j] = pBests[i-1][j]
                
        # gBest 更新
        temp_index = np.argmax(fitness[i])  # 評価値が最高の粒子の番号
        temp_value = np.max(fitness[i])     # 最高の評価値
        # 更新する
        if temp_value > gBests_fitness[i-1]:
            gBests_fitness[i] = temp_value
            gBests[i] = swarms[i][temp_index]
            gBests_paths[i] = paths[i][temp_index]
            gBests_bns[i] = bns[i][temp_index]
            gBests_sums[i] = sums[i][temp_index]
        # 更新しない
        else:
            gBests_fitness[i] = gBests_fitness[i-1]
            gBests[i] = gBests[i-1]
            gBests_paths[i] = gBests_paths[i-1]
            gBests_bns[i] = gBests_bns[i-1]
            gBests_sums[i] = gBests_sums[i-1]

    # 結果
    l = num_gen - 1
    print('<PSO>')
    print('BottleNeck: ', gBests_bns[l], 'path', gBests_paths[l])
    # print('gen', l, ', BottleNeck:', gBests_bns[l], ', path:', gBests_paths[l], ', sum:', gBests_sums[l])

    # シミュレーション関数の出力
    # ★変更：最終世代の値だけでなく、全世代のgBestボトルネック容量のリストを返す
    return gBests_bns, bn_inv

###############
### main関数 ###
###############
if __name__ == '__main__':
    num_simulation = 3  # シミュレーション回数
    num_nodes = 100   # ノード数
    num_par = 10    # 粒子数
    num_gen = 1000    # 世代数
    
    # 同じグラフに対して行う場合、以下
    # Graph = create_graph.ba_graph(num_node=num_nodes)
    # Graph_a = create_graph.add_optimal_path(Graph=Graph)
    
    # 出力
    result_path = savef.create_dir()    # 結果を格納するディレクトリのパス
    # シミュレーションの説明
    savef.make_doc(result_path, num_node=num_nodes, num_par=num_par, num_gen=num_gen)
    
    # ★変更：Long format（縦持ち）で結果を格納するリスト
    simulation_results_long = []
    
    # num_simulation回シミュレーションを行う
    for iter in range(num_simulation):
        print('試行回数:', iter)
        # 毎回グラフを生成
        Graph = create_graph.rnd_graph(num_node=num_nodes)
        
        # ★変更：シミュレーション結果（全世代のgBest履歴）を受け取る
        gBests_bns_history, bn_inv = simulation(Graph=Graph, gen=num_gen, par=num_par)
        
        # ★変更：シミュレーション結果をLong formatで格納
        # gBests_bns_history (リスト) をループ処理
        for gen_index, gbest_bn_value in enumerate(gBests_bns_history):
            simulation_results_long.append([iter, gen_index, gbest_bn_value, bn_inv])
    
    # ファイル名を生成
    # ★変更：ファイル名が履歴であることを示すように変更 (任意)
    file_name = datetime.now().strftime('%Y%m%d_%H%M%S') + '_gbest_history.csv'
    file_path = result_path + '/' + file_name
    
    # ファイルに書き出し
    with open(file_path, 'w', newline='') as f: # ★変更：newline='' を追加 (csvモジュールの推奨)
        writer = csv.writer(f)
        # ★変更：ヘッダーをLong formatに合わせて変更
        writer.writerow(['iter', 'generation', 'gBest_Bottleneck', 'inverse_dijkstra_bn'])
        # ★変更：Long formatのデータを書き出し
        writer.writerows(simulation_results_long)