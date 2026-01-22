# create_graph_4.py

from datetime import datetime
import networkx as nx
import random
import math

"""
Create graph object using NetworkX module.
This version creates graphs with 4 QoS attributes.
"""

# ★★★ 変更点：全グラフ生成関数に reliability と reliability_cost を追加 ★★★

def random_graph(num_node: int, prob: float = 0.2, lb_weight: int = 1, ub_weight: int = 50, lb_delay: float = 1.0, ub_delay: float = 10.0):
    """
    ランダムグラフを作成する関数。
    4つのQoS属性を持つエッジを追加する。
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_node))
    for i in range(num_node - 1):
        for j in range(i + 1, num_node):
            if random.random() < prob:
                G.add_edge(i, j)
    
    for (i,j) in G.edges():
        G[i][j]["weight"] = random.randint(lb_weight, ub_weight) # Bottleneck
        G[i][j]["delay"] = random.uniform(lb_delay, ub_delay) # Delay
        
        loss_rate = random.uniform(0.0001, 0.03) # 0.01% - 3%
        G[i][j]["loss_rate"] = loss_rate
        G[i][j]["loss_log_cost"] = -math.log(1 - loss_rate)
        
        reliability = random.uniform(0.99, 0.99999) # 99.0% - 99.999%
        G[i][j]['reliability'] = reliability
        G[i][j]['reliability_cost'] = -math.log(reliability)
    
    return G

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
        
        loss_rate = random.uniform(0.0001, 0.03)
        G[i][j]['loss_rate'] = loss_rate
        G[i][j]['loss_log_cost'] = -math.log(1 - loss_rate)
        
        reliability = random.uniform(0.99, 0.99999)
        G[i][j]['reliability'] = reliability
        G[i][j]['reliability_cost'] = -math.log(reliability)
        
    return G

def ba_graph(num_node, m=2, **kwargs):
    """
    BAモデル（スケールフリー）のネットワークを生成する関数
    """
    G = nx.barabasi_albert_graph(num_node, m)
    
    for (i, j) in G.edges():
        G[i][j]['weight'] = random.randint(1, 20)
        G[i][j]['delay'] = random.uniform(0.1, 1.0)
        
        loss_rate = random.uniform(0.0001, 0.03)
        G[i][j]['loss_rate'] = loss_rate
        G[i][j]['loss_log_cost'] = -math.log(1 - loss_rate)
        
        reliability = random.uniform(0.99, 0.99999)
        G[i][j]['reliability'] = reliability
        G[i][j]['reliability_cost'] = -math.log(reliability)
        
    return G