import time
import cProfile
import pstats
import io
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
import create_graph_4  # あなたのモジュール

# ==========================================
# 1. 計測対象の関数 (watase_34_multiprocessing.pyより抽出)
# ==========================================
def PathEncode(Particle, Graph, src, dst):
    path = [src]
    current_node = src
    visited = {src}
    limit_len = len(Particle)
    
    while current_node != dst:
        neighbors = [n for n in Graph.neighbors(current_node) if n not in visited]
        if not neighbors: return path, False
        
        best_neighbor = -1
        highest_prio = -1.0
        
        for neighbor in neighbors:
            if neighbor < limit_len:
                prio = Particle[neighbor]
                if prio > highest_prio: 
                    highest_prio = prio
                    best_neighbor = neighbor
                    
        if best_neighbor == -1: return path, False
        
        current_node = best_neighbor
        path.append(current_node)
        visited.add(current_node)
        
        if len(path) > limit_len: return path, False
        
    return path, True

# ==========================================
# 2. PSOと同じノード選択ロジック
# ==========================================
def select_nodes_like_pso(G):
    """
    watase_34_multiprocessing.py と全く同じロジックでsrc, dstを選ぶ
    """
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        nodes = list(largest_cc)
    else:
        nodes = list(G.nodes())
        
    # ノードが2つ未満ならエラー回避（実験用）
    if len(nodes) < 2:
        return nodes[0], nodes[0]
        
    return random.sample(nodes, 2)

# ==========================================
# 3. ベンチマーク実行関数
# ==========================================
def run_benchmark():
    # ★重要: 再現性のためシードを固定 (毎回同じランダムペアが選ばれるようにする)
    random.seed(42)
    np.random.seed(42)

    # ノード数の範囲
    node_counts = [100, 500]
    
    times_dense = []
    times_sparse = [] 
    
    print(f"=== ベンチマーク開始 (Nodes: {node_counts}) ===")

    for N in node_counts:
        # ----------------------------------------
        # 条件1: Dense Graph (現状: prob=0.2)
        # ----------------------------------------
        G_dense = create_graph_4.random_graph(num_node=N, prob=0.2)
        particle = np.random.uniform(0, 1, N)
        
        # ★PSOと同じ方法で選択
        src, dst = select_nodes_like_pso(G_dense)
        
        start = time.perf_counter()
        for _ in range(10):
            PathEncode(particle, G_dense, src, dst)
        avg_time = (time.perf_counter() - start) / 10
        times_dense.append(avg_time)
        
        # ----------------------------------------
        # 条件2: Sparse Graph (比較用: 次数固定)
        # ----------------------------------------
        prob_sparse = 10 / N 
        G_sparse = create_graph_4.random_graph(num_node=N, prob=prob_sparse)
        
        # ★PSOと同じ方法で選択
        src, dst = select_nodes_like_pso(G_sparse)
        
        start = time.perf_counter()
        for _ in range(10):
            PathEncode(particle, G_sparse, src, dst)
        avg_time = (time.perf_counter() - start) / 10
        times_sparse.append(avg_time)
        
        print(f"Nodes {N}: Dense={times_dense[-1]:.4f}s, Sparse={times_sparse[-1]:.4f}s")

    return node_counts, times_dense, times_sparse

# ==========================================
# 4. 解析と可視化
# ==========================================
def analyze_and_plot(nodes, t_dense, t_sparse):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左: 線形スケール
    ax1.plot(nodes, t_dense, 'o-', label='Dense (prob=0.2)', color='red')
    ax1.plot(nodes, t_sparse, 's-', label='Sparse (deg=10)', color='blue')
    ax1.set_title('Execution Time (Linear Scale)')
    ax1.set_xlabel('Number of Nodes (N)')
    ax1.set_ylabel('Time (sec)')
    ax1.legend()
    ax1.grid(True)
    
    # 右: 両対数スケール
    ax2.loglog(nodes, t_dense, 'o-', label='Dense (Current)', color='red')
    ax2.loglog(nodes, t_sparse, 's-', label='Sparse (Fixed Degree)', color='blue')
    
    # 補助線
    if t_sparse[0] > 0:
        base_y = t_sparse[0]
        base_x = nodes[0]
        y_n1 = [base_y * (n / base_x) for n in nodes]
        y_n2 = [base_y * (n / base_x)**2 for n in nodes]
        ax2.loglog(nodes, y_n1, '--', color='gray', label='O(N)')
        ax2.loglog(nodes, y_n2, ':', color='black', label='O(N^2)')
    
    ax2.set_title('Computational Complexity (Log-Log Scale)')
    ax2.set_xlabel('Number of Nodes (N)')
    ax2.set_ylabel('Time (sec)')
    ax2.legend()
    ax2.grid(True, which="both", ls="-")
    
    plt.tight_layout()
    plt.savefig('benchmark_result.png')
    print("\nグラフを 'benchmark_result.png' に保存しました。")

    # プロファイリング (Dense N=2000)
    print("\n=== 詳細プロファイリング (N=2000, Dense) ===")
    pr = cProfile.Profile()
    N = 2000
    G = create_graph_4.random_graph(N, prob=0.2)
    p = np.random.uniform(0, 1, N)
    src, dst = select_nodes_like_pso(G) # ここも同じロジックで
    
    pr.enable()
    PathEncode(p, G, src, dst)
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats(15)
    print(s.getvalue())

if __name__ == "__main__":
    nodes, t_dense, t_sparse = run_benchmark()
    analyze_and_plot(nodes, t_dense, t_sparse)