from datetime import datetime
import networkx as nx
import random

"""
Create graph object using NetworkX module
"""

# ランダムグラフ
def rnd_graph(num_node: int, prob: float = 0.2, lb: int = 1, ub: int = 100):
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
        RandGraph[i][j]["weight"] = random.randint(lb, ub)
    
    return RandGraph



# BAグラフ
def ba_graph(num_node: int, num_edge: int = 2, lb: int = 1, ub: int = 20):
    BAGraph = nx.barabasi_albert_graph(num_node, num_edge)
    for (i, j) in BAGraph.edges():
        BAGraph[i][j]["weight"] = random.randint(lb, ub)
        
    return BAGraph



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
    Graph[src][node1]["weight"] = 100
    Graph.add_edge(node1, node2)
    Graph[node1][node2]["weight"] = 100
    Graph.add_edge(node2, node3)
    Graph[node2][node3]["weight"] = 100
    Graph.add_edge(node3, dst)
    Graph[node3][dst]["weight"] = 100
    
    return Graph



# グラフを保存
def save_graph(Graph):
    file_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    nx.write_edgelist(Graph, file_name, comments='#', delimiter=' ', data=["weight"], encoding='utf-8')
    
    return file_name



# グラフを読み込み
def read_graph(file_name):
    Graph = nx.read_edgelist(file_name, nodetype=int, data=(("weight", int), ))
    
    return Graph

