import numpy as np
import random
import networkx as nx
from datetime import datetime
import csv
import savef

"""
GA simulation ver 1
    graph: BA
    
    delayの実装途中
"""

###########
### 関数 ###
###########

# BAグラフの生成
def ba_graph(num_node: int, num_edge: int = 2, lb: int = 1, ub: int = 20):
    BAGraph = nx.barabasi_albert_graph(num_node, num_edge)
    for (i, j) in BAGraph.edges():
        BAGraph[i][j]["bandwidth"] = random.randint(lb, ub)
        BAGraph[i][j]["delay"] = random.randint(lb, ub)
        
    return BAGraph

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
        # 隣接点がない場合
        if len(adjs) == 0:
            break
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

# 入力：経路
# 出力：ボトルネックリンク帯域
def path2bn(Graph, Vk):
    bn = 100
    for i in range(len(Vk)-1):
        temp = Graph[Vk[i]][Vk[i+1]]['bandwidth']
        if temp < bn:
            bn = temp
    
    return bn

# 入力：経路
# 出力：遅延
# 経路から重みを計算する関数
def path2delay(Graph, Vk):
    delay = 0
    print('Vk: ', Vk)
    for i in range(len(Vk) - 1):
        print(Vk[i], Vk[i+1])
        print(Graph[Vk[i]][Vk[i+1]]['delay'])
        print('')
        delay += Graph[Vk[i]][Vk[i+1]]['delay']
        
    return delay

# 手動で最適路を追加
def add_optimal_path(Graph, src, dst):
    # ノード数を取得
    num_node = len(Graph.nodes())
    # src -> 1 -> 2 -> 3 -> dst
    node_list = [i for i in range(0, num_node)] # ノードのリストを生成
    node_list.remove(src)   # 始点を除外
    node_list.remove(dst)   # 終点を除外
    optimal_nodes = random.sample(node_list, 3)
    node1 = optimal_nodes[0]
    node2 = optimal_nodes[1]
    node3 = optimal_nodes[2]
    
    # 経路と重みを追加
    Graph.add_edge(src, node1)
    Graph[src][node1]["bandwidth"] = 100
    Graph[src][node1]["delay"] = 100
    Graph.add_edge(node1, node2)
    Graph[node1][node2]["bandwidth"] = 100
    Graph[node1][node2]["delay"] = 100
    Graph.add_edge(node2, node3)
    Graph[node2][node3]["bandwidth"] = 100
    Graph[node2][node3]["delay"] = 100
    Graph.add_edge(node3, dst)
    Graph[node3][dst]["bandwidth"] = 100
    Graph[node3][dst]["delay"] = 100
    
    return Graph


# GAシミュレーションを行う関数
def GAsimulation(Graph, gen, chr, src_node, dst_node):
    # 定数
    num_node = int(len(Graph.nodes()))   # ノード数
    num_chr = chr   # 個体数
    num_gen = gen   # 世代数
    lb, ub = 1, 20  # 乱数の加減と上限

    ### 配列の宣言
    # 個体集団
    population = np.zeros((num_gen, num_chr, num_node))
    # 経路
    paths = [0] * num_gen
    # 経路妥当フラグ
    valids = [0] * num_gen
    # ボトルネックリンク帯域
    bns = np.zeros((num_gen, num_chr))
    # 遅延
    delays = np.zeros_like(bns)
    # 適応度
    fitness = np.zeros_like(bns)
    
    ### 初期化
    # 初期個体
    ini_pop = np.array([[random.randint(lb,ub) for _ in range(num_node)] for _ in range(num_chr)])
    # 初期個体を個体集団に格納
    population[0] = ini_pop

    # 初期個体を経路に
    path_list = []
    valid_list = []
    for chr in range(num_chr):
        path_tmp, valid_tmp = PathEncode(population[0][chr], Graph=BAGraph, src=src_node, dst=dst_node)
        path_list.append(path_tmp)
        valid_list.append(valid_tmp)
    paths[0] = path_list     # 初期個体の経路
    valids[0] = valid_list  # 初期個体の経路の妥当フラグ

    bns_list = []       # ボトルネックリンク帯域
    delays_list = []    # 遅延
    for chr in range(len(population[0])):
        bns_tmp = path2bn(BAGraph, paths[0][chr])
        # delays_tmp = path2delay(BAGraph, paths[0][chr])
        bns_list.append(bns_tmp)
        # delays_list.append(delays_tmp)
    bns[0] = bns_list
    # delays[0] = delays_list

    # 適応度の計算
    fitness[0] = bns[0]
    for chr in range(len(population[0])):
        if valids[0][chr] == False:
            fitness[0][chr] = 0
            
    # ランキング法で上位を選択
    # argsortで並べかえ
    sorted_indices = np.argsort(-fitness[0])
    num_sel = 3 # 選択する個体数

    p = np.zeros((num_sel, num_node))
    for i in range(num_sel):
        id = (sorted_indices[i])
        p[i] = population[0][id]

    # ランキング法で次世代の個体集団を生成
    # 選択された個体ひとつあたりの次の世代での個数
    num_devide = int(np.ceil(num_chr/num_sel))
    # 次の世代
    population_temp = np.zeros_like(ini_pop)

    for i in range(num_chr):
        id = int(np.floor(i / num_devide))
        population_temp[i] = p[id]

    # 一点交叉
    # 一点交叉を行う個体の割合
    p_c = 0.8
    # 次世代の個体集団の順番をシャッフル
    np.random.shuffle(population_temp)
    # 2個のペアずつ判定
    for i in range(1, len(population_temp), 2):
        if i >= len(population_temp) - 1:
            break
        elif random.random() <= p_c:
            # 確率で交叉が発生
            len_temp = random.randint(1, num_node)
            # i番目を分割
            temp1_head = population_temp[i][:len_temp].copy()
            temp1_tail = population_temp[i][len_temp:].copy()
            temp2_head = population_temp[i+1][:len_temp].copy()
            temp2_tail = population_temp[i+1][len_temp:].copy()
            population_temp[i] = np.concatenate([temp1_head, temp2_tail],0)
            population_temp[i+1] = np.concatenate([temp2_head, temp1_tail],0)

    # 突然変異
    # 対象：population_temp
    # 確率
    p_m = 0.01
    for p in range(num_chr):
        if random.random() >= p_m:
            id = random.randint(0,num_node-1)
            population_temp[p][id] = random.randint(1,20)

    # 選択・交叉・突然変異後の個体集団を次の世代に格納
    population[1] = population_temp
    
    ### 最適化
    for g in range(1, num_gen, 1):
        # 世代gの個体を経路に
        path_list = []
        valid_list = []
        for chr in range(num_chr):
            path_tmp, valid_tmp = PathEncode(population[g][chr], Graph=BAGraph, src=src_node, dst=dst_node)
            path_list.append(path_tmp)
            valid_list.append(valid_tmp)
        paths[g] = path_list    # 世代gの個体の経路
        valids[g] = valid_list  # 世代gの個体の経路の妥当フラグ

        bns_list = []       # ボトルネックリンク帯域
        delays_list = []    # 遅延
        for chr in range(len(population[g])):
            bns_tmp = path2bn(BAGraph, paths[g][chr])
            delays_tmp = path2delay(BAGraph, paths[g][chr])
            bns_list.append(bns_tmp)
            delays_list.append(delays_tmp)
        # ボトルネックリンク帯域の格納
        bns[g] = bns_list
        # 遅延の格納
        delays[g] = delays_list
        
        # 適応度の計算
        fitness[g] = bns[g]
        for chr in range(len(population[g])):
            if valids[g][chr] == False:
                fitness[g][chr] = 0
        
        ### 選択
        # ランキング法で上位を選択
        # argsortで並べかえ
        sorted_indices = np.argsort(-fitness[g])
        num_sel = 5 # 5個の個体を選択

        p = np.zeros((num_sel, num_node))
        for i in range(num_sel):
            id = (sorted_indices[i])
            p[i] = population[g][id]
        
        # ランキング法で次世代の個体集団を生成
        # 選択された個体ひとつあたりの次の世代での個数
        num_devide = int(np.ceil(num_chr/num_sel))
        # 次の世代
        population_temp = np.zeros_like(ini_pop)

        for i in range(num_chr):
            id = int(np.floor(i / num_devide))
            population_temp[i] = p[id]
        
        ### 交叉
        # 一点交叉
        # 一点交叉を行う個体の割合
        p_c = 0.8
        # 次世代の個体集団の順番をシャッフル
        np.random.shuffle(population_temp)
        # 2個のペアずつ判定
        for i in range(1, len(population_temp), 2):
            if i >= len(population_temp) - 1:
                break
            elif random.random() <= p_c:
                # 確率で交叉が発生
                len_temp = random.randint(1, num_node)
                # i番目を分割
                temp1_head = population_temp[i][:len_temp].copy()
                temp1_tail = population_temp[i][len_temp:].copy()
                temp2_head = population_temp[i+1][:len_temp].copy()
                temp2_tail = population_temp[i+1][len_temp:].copy()
                population_temp[i] = np.concatenate([temp1_head, temp2_tail],0)
                population_temp[i+1] = np.concatenate([temp2_head, temp1_tail],0)
        
        # 突然変異
        # 対象：population_temp
        # 確率
        p_m = 0.01
        for p in range(num_chr):
            if random.random() >= p_m:
                id = random.randint(0,num_node-1)
                population_temp[p][id] = random.randint(1,20)
                
        # 選択・交叉・突然変異後の個体集団を次の世代に格納
        if g + 1 >= num_gen:
            pass
        else:
            population[g + 1] = population_temp

    ### 結果
    # 上位個体の選択
    gen = num_gen - 1
    # argsortで並べかえ
    sorted_indices = np.argsort(-fitness[g])
    best_id = sorted_indices[0]
    
    print('<GA>')
    print('BottleNeck: ', bns[gen][best_id])
    print('path', paths[gen][best_id])
    # print('gen', l, ', BottleNeck:', gBests_bns[l], ', path:', gBests_paths[l], ', sum:', gBests_sums[l])

    # シミュレーション関数の出力
    return bns[gen][best_id]

### PSOシミュレーション
def PSOsimulation(Graph, gen, par, src_node, dst_node):
    # 定数
    num_node = int(len(Graph.nodes()))   # ノード数
    num_par = par   # 粒子数
    num_gen = gen   # 世代数
    lb, ub = 1, 20  # 乱数の加減と上限
    c_1 = 0.7       # PSOのパラメータ1: pBest
    c_2 = 0.3       # PSOのパラメータ2: gBest
    
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
    gBests_bns = [0] * num_gen

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
        bns_temp = path2bn(Graph, paths[0][j])
        bns_list.append(bns_temp)
    bns[0] = bns_list
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
            bns_temp = path2bn(Graph, paths[i][j])
            bns_list.append(bns_temp)
        bns[i] = bns_list
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
    print('BottleNeck: ', gBests_bns[l])
    print('path', gBests_paths[l])
    # print('gen', l, ', BottleNeck:', gBests_bns[l], ', path:', gBests_paths[l], ', sum:', gBests_sums[l])

    # シミュレーション関数の出力
    return gBests_bns[num_gen - 1]


###############
### main関数 ###
###############
if __name__ == '__main__':
    num_simulation = 100  # シミュレーション回数
    num_nodes = 50   # ノード数
    num_chr = 50    # 個体数
    num_gen = 50    # 世代数
    
    # 出力
    result_path = savef.create_dir()    # 結果を格納するディレクトリのパス
    # シミュレーションの説明
    savef.make_doc(result_path, num_node=num_nodes, num_par=num_chr, num_gen=num_gen)
    
    bottleneck_links = []   # シミュレーション結果のgBestのボトルネックリンク容量を格納するリスト(出力用)
    
    # num_simulation回シミュレーションを行う
    for iter in range(num_simulation):
        print('')
        print('試行回数:', iter)
        # BAグラフの生成
        # エッジが帯域 bandwidth と 遅延 delay をもつBAグラフ
        BAGraph = ba_graph(num_node=num_nodes)
        # 始点と終点の選択
        node_list = [i for i in range(0, num_nodes)]  # ノードのリストを生成
        src_node = random.choice(node_list)     # 始点の選択
        node_list.remove(src_node)              # 始点をリストから除外
        dst_node = random.choice(node_list)     # 終点の選択
        
        # 恣意的な最適経路を追加
        add_optimal_path(Graph=BAGraph, src=src_node, dst=dst_node)
        print('始点->終点: ' + str(src_node) + '->' + str(dst_node))
        
        # Dijkstra
        for (i,j) in BAGraph.edges():
            temp = 1/BAGraph[i][j]["bandwidth"]
            BAGraph[i][j]['inv'] = temp
        # 容量の逆数に対するDijkstraの実行
        v_inv = nx.dijkstra_path(BAGraph, source=src_node, target=dst_node, weight='inv')
        bn_inv = path2bn(BAGraph, v_inv)
        print('<Modified Dijkstra>')
        print('BottleNeck: ', bn_inv)
        print('path', v_inv)
        
        # GAシミュレーション
        GAresult = GAsimulation(Graph=BAGraph, gen=num_gen, chr=num_chr, src_node=src_node, dst_node=dst_node)
        
        # PSOシミュレーション
        PSOresult = PSOsimulation(Graph=BAGraph, gen=num_gen, par=num_chr, src_node=src_node, dst_node=dst_node)
        
        # シミュレーション結果を格納
        bottleneck_links.append([iter, bn_inv, GAresult, PSOresult])
    
    # ファイル名を生成
    file_name = 'GA' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv'
    file_path = result_path + '/' + file_name
    
    # ファイルに書き出し
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['iter', 'Dijkstra', 'GA', 'PSO'])
        writer.writerows(bottleneck_links)