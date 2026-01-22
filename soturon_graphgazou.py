import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# 図の保存用設定
def save_plot(filename, title):
    plt.title(title, fontsize=15, y=1.02)
    plt.axis('off')
    # PDFとPNGの両方で保存（論文にはPDFがおすすめ）
    plt.savefig(f"{filename}.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved: {filename}.pdf / .png")

# --- 図2.1: グラフの基本要素とパス・ウォーク ---
def draw_graph_def():
    G = nx.DiGraph()
    # ノードとエッジの定義
    edges = [('A', 'B', 3), ('B', 'C', 5), ('C', 'D', 2), ('B', 'D', 10), ('D', 'A', 4)]
    G.add_weighted_edges_from(edges)

    pos = nx.spring_layout(G, seed=42) # レイアウト固定
    
    plt.figure(figsize=(6, 4))
    
    # ノードとエッジの描画
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=2, arrowstyle='-|>', arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
    
    # 重みの表示
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    save_plot("graph_def", "Fig 2.1: Graph Definition (Path vs Walk)")

# --- 図2.2: SPP vs MBL ---
def draw_spp_mbl():
    G = nx.DiGraph()
    # 上のルート（MBL用: 帯域広いが遠い）
    G.add_edge('S', 'A', weight=10, bw=100) # weight=cost, bw=bandwidth
    G.add_edge('A', 'B', weight=10, bw=100)
    G.add_edge('B', 'G', weight=10, bw=100)
    
    # 下のルート（SPP用: 近いが帯域狭い）
    G.add_edge('S', 'C', weight=2, bw=10)
    G.add_edge('C', 'G', weight=2, bw=10)

    pos = {'S': (0, 1), 'A': (1, 2), 'B': (2, 2), 'G': (3, 1), 'C': (1.5, 0)}
    
    plt.figure(figsize=(8, 4))
    
    # エッジの色分け（上：青、下：赤）
    edges_mbl = [('S', 'A'), ('A', 'B'), ('B', 'G')]
    edges_spp = [('S', 'C'), ('C', 'G')]
    
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightgray', edgecolors='black')
    nx.draw_networkx_labels(G, pos)
    
    # MBL Path (Blue)
    nx.draw_networkx_edges(G, pos, edgelist=edges_mbl, width=3, edge_color='blue', label='MBL Path (Max BW=100)')
    # SPP Path (Red)
    nx.draw_networkx_edges(G, pos, edgelist=edges_spp, width=3, edge_color='red', style='dashed', label='SPP Path (Min Cost=4)')
    
    # ラベル描画（帯域とコスト）
    labels = {
        ('S', 'A'): "100Mbps\n(Cost 10)", ('A', 'B'): "100Mbps\n(Cost 10)", ('B', 'G'): "100Mbps\n(Cost 10)",
        ('S', 'C'): "10Mbps\n(Cost 2)", ('C', 'G'): "10Mbps\n(Cost 2)"
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)
    
    plt.legend(loc='lower right')
    save_plot("spp_vs_mbl", "Fig 2.2: SPP vs MBL Solutions")

# --- 図2.3: MCOP Concept ---
def draw_mcop_concept():
    plt.figure(figsize=(6, 5))
    
    # データの散布図（帯域 vs 遅延）
    bandwidths = [10, 20, 30, 80, 90, 100, 120, 50, 60]
    delays =     [ 5,  8, 12, 40, 45,  50,  60, 20, 25]
    
    # 制約ライン (Delay <= 30)
    D_max = 30
    
    # プロット
    for b, d in zip(bandwidths, delays):
        color = 'red' if d > D_max else 'green' # 制約違反は赤、OKは緑
        plt.scatter(b, d, color=color, s=100, edgecolors='black')

    # 制約領域の塗りつぶし
    plt.axhline(y=D_max, color='gray', linestyle='--', label=f'Delay Limit ($D_{{max}}={D_max}$)')
    plt.fill_between([0, 150], 0, D_max, color='green', alpha=0.1, label='Feasible Region')
    
    # 最適解の強調
    # Feasible (d <= 30) の中で Bandwidth最大を探す
    feasible = [(b, d) for b, d in zip(bandwidths, delays) if d <= D_max]
    best = max(feasible, key=lambda x: x[0])
    
    plt.annotate('Optimal Solution\n(Max BW in Feasible)', xy=best, xytext=(best[0]-40, best[1]+10),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.xlabel('Bottleneck Bandwidth (Mbps)', fontsize=12)
    plt.ylabel('Total Delay (ms)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    save_plot("mcop_concept", "Fig 2.3: Multi-Constrained Optimal Path (MCOP)")

# --- 実行 ---
if __name__ == "__main__":
    try:
        draw_graph_def()
        draw_spp_mbl()
        draw_mcop_concept()
        print("All diagrams generated successfully!")
    except ImportError as e:
        print("Required libraries not found. Please install: pip install networkx matplotlib numpy")