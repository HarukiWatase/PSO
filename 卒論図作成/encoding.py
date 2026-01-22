import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def draw_encoding_concept():
    # 1. グラフの定義 (論文の例に近いトポロジーを作成)
    G = nx.DiGraph()
    # ノード位置の定義 (可視化用)
    pos = {
        'S': (0, 1),
        '1': (1, 2), '2': (1, 0),
        '3': (2, 2), '4': (2, 0),
        'D': (3, 1)
    }
    
    # エッジの追加
    edges = [
        ('S', '1'), ('S', '2'),
        ('1', '3'), ('1', '4'),
        ('2', '3'), ('2', '4'),
        ('3', 'D'), ('4', 'D'),
        ('2', '1'), ('3', '4') # 相互接続など
    ]
    G.add_edges_from(edges)

    # 2. 優先度(Priority)の設定 (これが粒子の位置ベクトルに対応)
    # ここでは意図的に S -> 2 -> 3 -> D が選ばれるように設定
    priorities = {
        'S': 0.5,  # Start (値は関係ない)
        '1': 0.4,
        '2': 0.9,  # Sの隣接で最大 -> 2へ
        '3': 0.8,  # 2の隣接(1,3,4)で最大 -> 3へ
        '4': 0.2,
        'D': 1.0   # Goal
    }

    # 3. 描画設定
    fig = plt.figure(figsize=(12, 6))
    
    # --- 左側: 優先度リスト (Particle Position) ---
    ax_table = fig.add_subplot(1, 3, 1)
    ax_table.axis('off')
    ax_table.set_title("1. Particle Position (Priority List)", fontsize=12, pad=20)
    
    # テーブルデータの作成
    table_data = [[node, f"{val:.1f}"] for node, val in priorities.items()]
    col_labels = ["Node ID", "Priority"]
    
    # テーブル描画
    table = ax_table.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # --- 中央: 矢印 ---
    ax_arrow = fig.add_subplot(1, 3, 2)
    ax_arrow.axis('off')
    ax_arrow.text(0.5, 0.5, "Decoding\n(Greedy Search)", ha='center', va='center', fontsize=14, weight='bold')
    ax_arrow.annotate("", xy=(1, 0.5), xytext=(0, 0.5), arrowprops=dict(arrowstyle="->", lw=2))

    # --- 右側: 経路選択の結果 (Network Graph) ---
    ax_graph = fig.add_subplot(1, 3, 3)
    ax_graph.set_title("2. Generated Path", fontsize=12)
    
    # ノードの描画 (優先度で色分け)
    node_colors = ['lightgray' for _ in G.nodes()]
    node_sizes = 800
    
    # 経路のハイライト (S -> 2 -> 3 -> D)
    path_nodes = ['S', '2', '3', 'D']
    path_edges = [('S', '2'), ('2', '3'), ('3', 'D')]
    
    # ノードラベル (ID : Priority)
    labels = {n: f"{n}\n({priorities[n]:.1f})" for n in G.nodes()}
    
    # グラフ描画
    # ベースの描画
    nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_size=node_sizes, node_color='white', edgecolors='black')
    nx.draw_networkx_edges(G, pos, ax=ax_graph, edge_color='gray', arrows=True, arrowstyle='-|>', arrowsize=15)
    
    # ハイライト経路の描画
    nx.draw_networkx_nodes(G, pos, ax=ax_graph, nodelist=path_nodes, node_size=node_sizes, node_color='lightblue', edgecolors='black')
    nx.draw_networkx_edges(G, pos, ax=ax_graph, edgelist=path_edges, edge_color='blue', width=2.5, arrows=True, arrowstyle='-|>', arrowsize=20)
    
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax_graph, font_size=10)

    # 解説テキスト
    info_text = (
        "Logic:\n"
        "1. Current: S\n"
        "   Neighbors: 1(0.4), 2(0.9)\n"
        "   Select 2 (Max Priority)\n\n"
        "2. Current: 2\n"
        "   Neighbors: 1(0.4), 3(0.8), 4(0.2)\n"
        "   Select 3 (Max Priority)\n\n"
        "3. Current: 3\n"
        "   Neighbors: D(1.0)\n"
        "   Select D (Goal)"
    )
    plt.figtext(0.75, 0.05, info_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig("encoding_concept.png", dpi=300)
    plt.show()

# 実行
draw_encoding_concept()