import networkx as nx
import numpy as np
import random

"""
GA simulation: baseline implementation without valid outputs.
"""

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
    for i in range(len(Vk) - 1):
        delay += Graph[Vk[i]][Vk[i+1]]['delay']
        
    return delay

# BAグラフの生成
n_n = 20 # ノード数
# エッジが帯域 bandwidth と 遅延 delay をもつBAグラフ
BAGraph = ba_graph(num_node=n_n)
# 始点と終点の選択
node_list = [i for i in range(0, n_n)]  # ノードのリストを生成
src_node = random.choice(node_list)     # 始点の選択
node_list.remove(src_node)              # 始点をリストから除外
dst_node = random.choice(node_list)     # 終点の選択

# 容量の逆数に対するDijkstraの実行
# inverse Dijkstra
for (i,j) in BAGraph.edges():
    temp = 1/BAGraph[i][j]["bandwidth"]
    BAGraph[i][j]['inv'] = temp
for i in BAGraph.nodes():
    for j in BAGraph.nodes():
        if i == j:
            break
        v_inv = nx.dijkstra_path(BAGraph, source=i, target=j, weight='inv')
        bn_inv = path2bn(BAGraph, v_inv)
        print('<Modified Dijkstra>')
        print('src: ', i)
        print('dst: ', j)
        print('BottleNeck: ', bn_inv, 'path', v_inv)

### GA
# GAのパラメータ
num_node = int(len(BAGraph.nodes())) # ノード数
num_chr = 100   # 個体数
num_gen = 100   # 世代数
lb = 1
ub = 20

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
    delays_tmp = path2delay(BAGraph, paths[0][chr])
    bns_list.append(bns_tmp)
    delays_list.append(delays_tmp)
bns[0] = bns_list
delays[0] = delays_list

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
    # ランキング法で上位2位を選択
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

### 出力
with open('GA/result.txt', 'w') as f:
    print('population', file=f)
    for i in range(num_gen):
        print('gen', i, file=f)
        for j in range(num_chr):
            print('chormosome', j, file=f)
            print(population[i][j], file=f)
            print('Bottleneck: ', bns[i][j], file=f)

# np.savetxt('GA/population.txt', population)
# np.savetxt('GA/path.txt', paths)
# np.savetxt('GA/bns.txt', bns)
# np.savetxt('GA/delay.txt', delays)
# np.savetxt('GA/fitness.txt', fitness)
