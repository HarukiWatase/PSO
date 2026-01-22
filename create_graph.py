from datetime import datetime
import networkx as nx
import random
import math

"""
Create graph object using NetworkX module
"""

# ランダムグラフ
def rnd_graph(num_node: int, prob: float = 0.2, lb_weight: int = 1, ub_weight: int = 20, lb_delay: float = 1.0, ub_delay: float = 10.0):
    """
    ランダムグラフを作成する関数。
    各エッジに 'weight' (容量) と 'delay' を追加する。
    Args:
        num_node (int): ノード数
        prob (float): エッジが存在する確率
        lb_weight (int): 重み (容量) の最小値
        ub_weight (int): 重み (容量) の最大値
        lb_delay (float): 遅延の最小値
        ub_delay (float): 遅延の最大値
    Returns:
        nx.Graph: 作成されたグラフオブジェクト
    """
    nodes = list(range(num_node))
    edges = []
    for i in range(num_node - 1):
        for j in range(i + 1, num_node):
            if random.random() < prob:
                edges.append((i, j))
    RandGraph = nx.Graph()
    RandGraph.add_nodes_from(nodes)
    RandGraph.add_edges_from(edges)
    
    for (i,j) in RandGraph.edges():
        RandGraph[i][j]["weight"] = random.randint(lb_weight, ub_weight)
        RandGraph[i][j]["delay"] = random.uniform(lb_delay, ub_delay) # ★追加: 遅延属性
    
    return RandGraph



def grid_graph(num_node, **kwargs):
    """
    グリッド状のネットワークを生成する関数
    """
    # num_nodeに最も近い平方数になるように格子サイズを決定
    m = int(math.sqrt(num_node))
    
    # ノード数がm*mになるようにグリッドグラフを生成
    G = nx.grid_2d_graph(m, m)
    
    # ノード名が(x,y)タプルになっているので、0からの連番整数に変換
    # これにより、他の関数との互換性が保たれる
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    # 各エッジにランダムな重み（ボトルネック）と遅延を付与
    for (i, j) in G.edges():
        G[i][j]['weight'] = random.randint(1, 20)
        G[i][j]['delay'] = random.uniform(0.1, 1.0)
        
    return G

def ba_graph(num_node, m=2, **kwargs):
    """
    BAモデル（スケールフリー）のネットワークを生成する関数
    m: 新しいノードが接続する既存ノードの数（次数に影響）
    """
    # 指定されたノード数とパラメータmでBAモデルのグラフを生成
    G = nx.barabasi_albert_graph(num_node, m)
    
    # 各エッジにランダムな重み（ボトルネック）と遅延を付与
    for (i, j) in G.edges():
        G[i][j]['weight'] = random.randint(1, 20)
        G[i][j]['delay'] = random.uniform(0.1, 1.0)
        
    return G

# 手動で最適路を追加
def add_optimal_path(Graph, src, dst, optimal_weight: int = 100, optimal_delay: float = 1.0):
    """
    グラフに手動で最適パスを追加する関数。
    追加するエッジに 'weight' と 'delay' を設定する。
    Args:
        Graph (nx.Graph): 対象のグラフオブジェクト
        src (int): 始点ノード
        dst (int): 終点ノード
        optimal_weight (int): 最適パスのリンク容量
        optimal_delay (float): 最適パスのリンク遅延
    Returns:
        nx.Graph: 最適パスが追加されたグラフオブジェクト
    """
    num_node = len(Graph.nodes())
    node_list = [i for i in range(0, num_node)]
    
    # 始点と終点がノードリストに存在するか確認し、存在しない場合は追加
    if src not in node_list:
        Graph.add_node(src)
        node_list.append(src)
    if dst not in node_list:
        Graph.add_node(dst)
        node_list.append(dst)

    # 始点と終点を除外して、中間ノードをランダムに3つ選択
    # 選択できるノードが不足している場合を考慮
    available_nodes = [n for n in node_list if n != src and n != dst]
    if len(available_nodes) < 3:
        # 中間ノードが不足している場合は、シンプルなパスにするかエラーを出すか検討
        # ここでは、可能な限りシンプルなパスにする (始点->中間ノード->終点)
        optimal_nodes = random.sample(available_nodes, min(len(available_nodes), 1))
    else:
        optimal_nodes = random.sample(available_nodes, 3)

    path_nodes = [src] + optimal_nodes + [dst]
    
    # 経路と重み、遅延を追加
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i+1]
        Graph.add_edge(u, v)
        Graph[u][v]["weight"] = optimal_weight
        Graph[u][v]["delay"] = optimal_delay # ★追加: 遅延属性
    
    return Graph


# グラフを保存
def save_graph(Graph, file_path): # file_path を引数に追加
    """
    グラフをエッジリスト形式で保存する関数。
    Args:
        Graph (nx.Graph): 保存するグラフオブジェクト
        file_path (str): 保存するファイルのフルパス (ディレクトリ含む)
    """
    # この関数は現在、sim241014_lts.py では直接使用されていませんが、
    # グラフのデバッグや再利用のために有用です。
    nx.write_edgelist(Graph, file_path, comments='#', delimiter=' ', data=["weight", "delay"], encoding='utf-8') # ★修正: delayも保存

# グラフを読み込み
def read_graph(file_name):
    """
    エッジリスト形式のグラフを読み込む関数。
    Args:
        file_name (str): 読み込むファイルのパス
    Returns:
        nx.Graph: 読み込まれたグラフオブジェクト
    """
    # weightとdelayの両方を読み込むように修正
    Graph = nx.read_edgelist(file_name, nodetype=int, data=(("weight", int), ("delay", float))) # ★修正: delayも読み込み
    
    return Graph