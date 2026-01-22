# create_graph.py

from datetime import datetime
import networkx as nx
import random
import math

"""
Create graph object using NetworkX module
"""

# ★★★ 変更点：全グラフ生成関数に loss_rate と loss_log_cost を追加 ★★★

def rnd_graph(num_node: int, prob: float = 0.2, lb_weight: int = 1, ub_weight: int = 20, lb_delay: float = 1.0, ub_delay: float = 10.0):
    """
    ランダムグラフを作成する関数。
    各エッジに 'weight', 'delay', 'loss_rate', 'loss_log_cost' を追加する。
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
        RandGraph[i][j]["delay"] = random.uniform(lb_delay, ub_delay)
        loss_rate = random.uniform(0.0001, 0.03) # 0.01% - 3% のロス率
        RandGraph[i][j]["loss_rate"] = loss_rate
        RandGraph[i][j]["loss_log_cost"] = -math.log(1 - loss_rate)
    
    return RandGraph

def grid_graph(num_node, **kwargs):
    """
    グリッド状のネットワークを生成する関数
    """
    m = int(math.sqrt(num_node))
    G = nx.grid_2d_graph(m, m)
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    for (i, j) in G.edges():
        G[i][j]['weight'] = random.randint(1, 20)
        G[i][j]['delay'] = random.uniform(0.1, 1.0)
        loss_rate = random.uniform(0.0001, 0.03) # 0.01% - 3% のロス率
        G[i][j]['loss_rate'] = loss_rate
        G[i][j]['loss_log_cost'] = -math.log(1 - loss_rate)
        
    return G

def ba_graph(num_node, m=2, **kwargs):
    """
    BAモデル（スケールフリー）のネットワークを生成する関数
    """
    G = nx.barabasi_albert_graph(num_node, m)
    
    for (i, j) in G.edges():
        G[i][j]['weight'] = random.randint(1, 20)
        G[i][j]['delay'] = random.uniform(0.1, 1.0)
        loss_rate = random.uniform(0.0001, 0.03) # 0.01% - 3% のロス率
        G[i][j]['loss_rate'] = loss_rate
        G[i][j]['loss_log_cost'] = -math.log(1 - loss_rate)
        
    return G

# (add_optimal_path, save_graph, read_graph は変更なし)
def add_optimal_path(Graph, src, dst, optimal_weight: int = 100, optimal_delay: float = 1.0):
    num_node = len(Graph.nodes())
    node_list = [i for i in range(0, num_node)]
    if src not in node_list:
        Graph.add_node(src)
        node_list.append(src)
    if dst not in node_list:
        Graph.add_node(dst)
        node_list.append(dst)
    available_nodes = [n for n in node_list if n != src and n != dst]
    if len(available_nodes) < 3:
        optimal_nodes = random.sample(available_nodes, min(len(available_nodes), 1))
    else:
        optimal_nodes = random.sample(available_nodes, 3)
    path_nodes = [src] + optimal_nodes + [dst]
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i+1]
        Graph.add_edge(u, v)
        Graph[u][v]["weight"] = optimal_weight
        Graph[u][v]["delay"] = optimal_delay
        # ★★★ 注意：手動追加パスのロス率は簡単のため0に設定 ★★★
        Graph[u][v]["loss_rate"] = 0.0
        Graph[u][v]["loss_log_cost"] = 0.0
    return Graph

def save_graph(Graph, file_path):
    # 将来使う場合のために data に loss_rate を追加
    nx.write_edgelist(Graph, file_path, comments='#', delimiter=' ', data=["weight", "delay", "loss_rate"], encoding='utf-8')

def read_graph(file_name):
    Graph = nx.read_edgelist(file_name, nodetype=int, data=(("weight", int), ("delay", float), ("loss_rate", float)))
    # 読み込んだ後で loss_log_cost を計算
    for i, j, data in Graph.edges(data=True):
        if "loss_rate" in data:
            Graph[i][j]['loss_log_cost'] = -math.log(1 - data['loss_rate'])
        else:
            Graph[i][j]['loss_log_cost'] = 0.0
    return Graph