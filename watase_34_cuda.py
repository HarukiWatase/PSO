# watase_34_cuda.py
# 目的: MCOP/MBL問題に対し、Numba CUDA + CuPyを用いてPSOを高速化し、厳密解法と比較する

import networkx as nx
import numpy as np
import os
import sys

# CUDA 11.x/12.xのbinディレクトリをPATHに自動追加（NVRTC DLLを見つけるため）
# 重要: CuPyがCUDA 11.8を使用するため、CUDA 11.xを優先的に検索
cuda_bin_paths = [
    # CUDA 11.x（優先）
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin",
    # CUDA 12.x（フォールバック）
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.7\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
]

current_path = os.environ.get('PATH', '')
cuda_path_found = None
cuda_11x_bin_path = None

# まずCUDA 11.xを探す（優先）
for cuda_bin_path in cuda_bin_paths:
    if os.path.exists(cuda_bin_path) and 'v11.' in cuda_bin_path:
        cuda_11x_bin_path = cuda_bin_path
        # PATHに追加（最優先）
        if cuda_bin_path not in current_path:
            os.environ['PATH'] = cuda_bin_path + os.pathsep + os.environ.get('PATH', '')
        # CUDA_PATH環境変数も設定
        cuda_versioned_path = os.path.dirname(cuda_bin_path)  # v11.8ディレクトリ
        os.environ['CUDA_PATH'] = cuda_versioned_path
        # Windows DLL検索用にbinディレクトリも追加（CuPyのインポート前に必要）
        try:
            os.add_dll_directory(cuda_bin_path)
        except Exception as e:
            print(f"[WARNING] Failed to add DLL directory: {e}")
        print(f"[INFO] CUDA 11.x detected: {cuda_bin_path}")
        print(f"[INFO] CUDA_PATH set to: {cuda_versioned_path}")
        break  # CUDA 11.xが見つかったら優先的に使用

# CUDA 11.xが見つからない場合、CUDA 12.xを使用
if cuda_11x_bin_path is None:
    for cuda_bin_path in cuda_bin_paths:
        if os.path.exists(cuda_bin_path):
            if cuda_bin_path not in current_path:
                os.environ['PATH'] = cuda_bin_path + os.pathsep + os.environ.get('PATH', '')
            cuda_path_found = cuda_bin_path
            try:
                os.add_dll_directory(cuda_bin_path)
            except Exception as e:
                print(f"[WARNING] Failed to add DLL directory: {e}")
            print(f"[INFO] CUDA detected: {cuda_bin_path}")
            break

# CuPyの初期化（NVRTCエラーを適切に処理、CUDA 11対応）
# 重要: CuPyのインポート前に、CUDA 11.8のbinディレクトリを確実にPATHに追加
CUDA_AVAILABLE = False
cp = np  # デフォルトはNumPy

# CuPyがCUDA 11.8のDLLを見つけられるように、環境変数を設定
if cuda_11x_bin_path and os.path.exists(cuda_11x_bin_path):
    # PATHの先頭に追加（最優先）
    current_path_env = os.environ.get('PATH', '')
    if cuda_11x_bin_path not in current_path_env:
        os.environ['PATH'] = cuda_11x_bin_path + os.pathsep + current_path_env
    
    # CUDA_PATH_V11_8も設定（CuPyがCUDA 11.8を優先的に使用するように）
    cuda_11x_root = os.path.dirname(cuda_11x_bin_path)
    os.environ['CUDA_PATH_V11_8'] = cuda_11x_root

# CuPyのインポート（NVRTC DLLの読み込みエラーを警告として処理）
try:
    import warnings
    # CuPyのインポート時の警告を一時的に抑制
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', message='.*nvrtc.*', category=RuntimeWarning)
        import cupy as cp
except ImportError:
    print("Warning: CuPy not found. Falling back to NumPy (CPU).")
    print("To install CuPy for CUDA 11.x, use: pip install cupy-cuda11x")
    CUDA_AVAILABLE = False
    cp = np  # Fallback to NumPy
except Exception as e:
    # CuPyのインポート時に発生する可能性のあるその他のエラー
    error_msg = str(e).lower()
    if 'nvrtc' in error_msg or 'dll' in error_msg:
        # NVRTC DLLのエラーは警告として表示するが、CuPyの使用を試みる
        print(f"Warning: CuPy import NVRTC DLL warning (may be non-critical): {e}")
        try:
            # CuPyを再インポートしてテスト
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                import cupy as cp
            # CuPyが実際に動作するか確認
            test_array = cp.array([1.0], dtype=cp.float32)
            result = test_array + 1.0
            _ = cp.asnumpy(result)
            del test_array, result
            CUDA_AVAILABLE = True
            print("CuPy initialized successfully despite NVRTC warning. GPU acceleration enabled.")
            print("Note: Some advanced features may be limited without NVRTC DLL.")
        except Exception as e2:
            print(f"Warning: CuPy initialization failed: {e2}")
            print("Falling back to NumPy (CPU) for array operations.")
            CUDA_AVAILABLE = False
            cp = np  # Fallback to NumPy
    else:
        print(f"Warning: CuPy import failed: {e}")
        print("Falling back to NumPy (CPU) for array operations.")
        CUDA_AVAILABLE = False
        cp = np  # Fallback to NumPy

# CuPyがインポートできた場合、動作確認
if 'cp' in locals() and cp is not np:
    try:
        # CUDA 11対応: CuPyのバージョンとCUDAバージョンを確認
        cupy_version = cp.__version__
        print(f"CuPy version: {cupy_version}")
        
        if hasattr(cp, 'cuda'):
            try:
                # GPUデバイス情報を取得
                device = cp.cuda.Device()
                compute_capability = device.compute_capability
                # CUDA Runtimeバージョンも取得
                try:
                    cuda_runtime_version = cp.cuda.runtime.runtimeGetVersion()
                    major = cuda_runtime_version // 1000
                    minor = (cuda_runtime_version % 1000) // 10
                    print(f"CUDA Runtime version: {major}.{minor}")
                except:
                    pass
                print(f"GPU Device: {device.id}, Compute Capability: {compute_capability}")
                
                # GPUメモリ情報も表示
                try:
                    meminfo = cp.cuda.runtime.memGetInfo()
                    print(f"GPU Memory: {meminfo[0] / 1024**3:.2f} GB free / {meminfo[1] / 1024**3:.2f} GB total")
                except:
                    pass
            except:
                pass
        
        # 簡単な操作でCuPyが動作するか確認
        test_array = cp.array([1, 2, 3])
        _ = test_array + 1
        # メモリ操作もテスト
        test_gpu = cp.zeros((10, 10), dtype=cp.float32)
        _ = test_gpu + 1.0
        del test_array, test_gpu
        CUDA_AVAILABLE = True
        print("CuPy initialized successfully. GPU acceleration enabled.")
    except Exception as e:
        # NVRTCエラーなどの実行時エラーをキャッチ
        # 重要: NVRTC DLLのエラーは無視して、CuPyが基本的な操作を実行できるか確認
        error_msg = str(e).lower()
        if 'nvrtc' in error_msg or 'dll' in error_msg:
            # NVRTC DLLのエラーは警告として表示するが、CuPyの使用を試みる
            print(f"Warning: CuPy NVRTC DLL warning (may be non-critical): {e}")
            # CuPyが実際に動作するか再確認（基本的な操作のみ）
            try:
                # より基本的な操作でテスト
                test_array = cp.array([1.0], dtype=cp.float32)
                result = test_array + 1.0
                _ = cp.asnumpy(result)  # CPUに転送して確認
                del test_array, result
                CUDA_AVAILABLE = True
                print("CuPy initialized successfully despite NVRTC warning. GPU acceleration enabled.")
                print("Note: Some advanced features may be limited without NVRTC DLL.")
            except Exception as e2:
                print(f"Warning: CuPy basic operations also failed: {e2}")
                print("Falling back to NumPy (CPU) for array operations.")
                CUDA_AVAILABLE = False
                cp = np  # Fallback to NumPy
        else:
            print(f"Warning: CuPy runtime error occurred: {e}")
            print("Falling back to NumPy (CPU) for array operations.")
            CUDA_AVAILABLE = False
            cp = np  # Fallback to NumPy

try:
    from numba import cuda
    from numba.cuda import random as cuda_random
    NUMBA_CUDA_AVAILABLE = True
    print("Numba CUDA detected. GPU kernels enabled.")
except ImportError:
    print("Warning: Numba CUDA not found. Falling back to CPU.")
    NUMBA_CUDA_AVAILABLE = False

import create_graph_4 as create_graph
import savef
from datetime import datetime
import csv
from heapq import heappop, heappush
import time
import math
import random

# ==========================================
# 1. GPU検出・初期化処理
# ==========================================

def init_gpu():
    """GPUデバイス情報を表示"""
    if NUMBA_CUDA_AVAILABLE:
        try:
            device = cuda.get_current_device()
            print(f"GPU Device: {device.name}")
            print(f"Compute Capability: {device.compute_capability}")
            if CUDA_AVAILABLE:
                try:
                    meminfo = cp.cuda.runtime.memGetInfo()
                    print(f"GPU Memory: {meminfo[0] / 1024**3:.2f} GB free / {meminfo[1] / 1024**3:.2f} GB total")
                except:
                    print("Warning: CuPy memory info not available")
            return True
        except Exception as e:
            print(f"Warning: GPU initialization failed: {e}. Using CPU fallback.")
            return False
    return False

# ==========================================
# 2. グラフ構造の隣接リスト形式変換（最適化版）
# ==========================================

def graph_to_adjacency_list(G):
    """
    NetworkXグラフを隣接リスト形式に変換（無向グラフを自然に扱える）
    Returns:
        adj_list: 各ノードの隣接ノードリスト（フラット配列）
        adj_ptr: 各ノードの隣接リストの開始インデックス (num_nodes + 1)
        edge_attrs: エッジ属性の辞書 (u, v) -> (weight, delay, loss_log_cost, reliability_cost)
    """
    start_time = time.time()
    num_nodes = len(G.nodes())
    
    # 隣接リストを構築（NetworkXが自動的に無向グラフを処理）
    t1 = time.time()
    adj_lists = [[] for _ in range(num_nodes)]
    edge_attrs = {}
    
    for u, v in G.edges():
        # 無向グラフなので、両方向を追加
        adj_lists[u].append(v)
        adj_lists[v].append(u)
        
        # エッジ属性を取得（一度だけ）
        edge_data = G.get_edge_data(u, v, {})
        edge_attrs[(u, v)] = (
            edge_data.get('weight', 0.0),
            edge_data.get('delay', float('inf')),
            edge_data.get('loss_log_cost', float('inf')),
            edge_data.get('reliability_cost', float('inf'))
        )
        # 逆方向も同じ属性
        edge_attrs[(v, u)] = edge_attrs[(u, v)]
    t2 = time.time()
    
    # フラット配列に変換
    t3 = time.time()
    adj_list = []
    adj_ptr = np.zeros(num_nodes + 1, dtype=np.int32)
    edge_weights = []
    edge_delays = []
    edge_loss_log_costs = []
    edge_reliability_costs = []
    
    edge_id = 0
    for u in range(num_nodes):
        adj_ptr[u] = edge_id
        for v in adj_lists[u]:
            adj_list.append(v)
            w, d, l, r = edge_attrs[(u, v)]
            edge_weights.append(w)
            edge_delays.append(d)
            edge_loss_log_costs.append(l)
            edge_reliability_costs.append(r)
            edge_id += 1
    adj_ptr[num_nodes] = edge_id
    t4 = time.time()
    
    # NumPy配列に変換
    t5 = time.time()
    adj_list = np.array(adj_list, dtype=np.int32)
    edge_weights = np.array(edge_weights, dtype=np.float32)
    edge_delays = np.array(edge_delays, dtype=np.float32)
    edge_loss_log_costs = np.array(edge_loss_log_costs, dtype=np.float32)
    edge_reliability_costs = np.array(edge_reliability_costs, dtype=np.float32)
    t6 = time.time()
    
    total_time = time.time() - start_time
    print(f"[PROFILE] graph_to_adjacency_list: total={total_time:.3f}s "
          f"(edges_loop={t2-t1:.3f}s, flatten={t4-t3:.3f}s, numpy={t6-t5:.3f}s)")
    
    return (adj_list, adj_ptr, edge_weights, edge_delays, 
            edge_loss_log_costs, edge_reliability_costs)

# ==========================================
# 3. Numba CUDAカーネル実装
# ==========================================

# PathEncode CUDAカーネル
if NUMBA_CUDA_AVAILABLE:
    @cuda.jit
    def path_encode_kernel(particles, row_ptr, col_idx, src, dst, num_nodes, 
                           limit_len, max_path_len, paths, path_lengths, is_valid):
        """
        PathEncodeをCUDAカーネルで実装
        1粒子=1スレッドで並列実行
        limit_len: 粒子の次元数（元のコードのlimit_len = len(Particle)）
        最適化: ビットマスクを使用してO(1)の訪問済みチェックを実装
        """
        particle_id = cuda.grid(1)
        num_particles = particles.shape[0]
        
        if particle_id >= num_particles:
            return
        
        # ビットマスク配列で訪問済みノードを管理（O(1)チェック）
        # num_nodesに基づいて動的にサイズを計算（最大2048ノードまで対応）
        bits_array_size = (num_nodes + 31) // 32  # 切り上げ
        if bits_array_size > 64:  # 64 * 32 = 2048ノードまで
            bits_array_size = 64
        visited_bits = cuda.local.array(64, dtype=np.int32)  # 最大2048ノードまで
        # ビットマスクを初期化（すべて0）
        for i in range(bits_array_size):
            visited_bits[i] = 0
        
        # ローカル配列で経路を管理
        path = cuda.local.array(1024, dtype=np.int32)  # 最大経路長
        
        # 初期化
        current_node = src
        path[0] = src
        path_len = 1
        # srcを訪問済みにマーク
        src_idx = src // 32
        src_bit = src % 32
        if src_idx < bits_array_size:
            visited_bits[src_idx] |= (1 << src_bit)
        
        # 経路探索
        while current_node != dst and path_len < max_path_len:
            # 隣接ノードを取得（CSR形式から）
            start_idx = row_ptr[current_node]
            end_idx = row_ptr[current_node + 1]
            
            best_neighbor = -1
            highest_prio = -1.0
            
            # 隣接ノードを走査
            for i in range(start_idx, end_idx):
                neighbor = col_idx[i]
                
                # 訪問済みチェック（O(1)ビットマスク操作）
                neighbor_idx = neighbor // 32
                neighbor_bit = neighbor % 32
                if neighbor_idx >= bits_array_size:
                    continue  # 範囲外はスキップ（通常は発生しない）
                if (visited_bits[neighbor_idx] & (1 << neighbor_bit)) != 0:
                    continue  # 訪問済み
                
                # 優先度チェック（CPU版と完全一致）
                if neighbor < limit_len:
                    prio = particles[particle_id, neighbor]
                    if prio > highest_prio:
                        highest_prio = prio
                        best_neighbor = neighbor
            
            if best_neighbor == -1:
                # 経路が見つからない
                is_valid[particle_id] = 0
                path_lengths[particle_id] = path_len
                for i in range(path_len):
                    paths[particle_id, i] = path[i]
                return
            
            # 次のノードに移動
            current_node = best_neighbor
            path[path_len] = current_node
            path_len += 1
            # 訪問済みにマーク
            best_idx = best_neighbor // 32
            best_bit = best_neighbor % 32
            if best_idx < bits_array_size:
                visited_bits[best_idx] |= (1 << best_bit)
        
        # 結果を出力
        if current_node == dst:
            is_valid[particle_id] = 1
        else:
            is_valid[particle_id] = 0
        
        path_lengths[particle_id] = path_len
        for i in range(path_len):
            paths[particle_id, i] = path[i]
        # 残りを-1で埋める
        for i in range(path_len, max_path_len):
            paths[particle_id, i] = -1

    # 属性計算CUDAカーネル
    @cuda.jit
    def calculate_attributes_kernel(paths, path_lengths, is_valid, row_ptr, col_idx,
                                    edge_weights, edge_delays, edge_loss_log_costs, 
                                    edge_reliability_costs, num_nodes,
                                    bottlenecks, delays, loss_rates, reliabilities):
        """
        経路属性を並列計算
        """
        particle_id = cuda.grid(1)
        num_particles = paths.shape[0]
        
        if particle_id >= num_particles:
            return
        
        if is_valid[particle_id] == 0:
            bottlenecks[particle_id] = 0.0
            delays[particle_id] = float('inf')
            loss_rates[particle_id] = 1.0
            reliabilities[particle_id] = 0.0
            return
        
        path_len = path_lengths[particle_id]
        if path_len < 2:
            bottlenecks[particle_id] = 0.0
            delays[particle_id] = float('inf')
            loss_rates[particle_id] = 1.0
            reliabilities[particle_id] = 0.0
            return
        
        # エッジ属性を集約
        min_bottleneck = float('inf')
        total_delay = 0.0
        total_loss_log_cost = 0.0
        total_reliability_cost = 0.0
        
        for i in range(path_len - 1):
            u = paths[particle_id, i]
            v = paths[particle_id, i + 1]
            
            if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
                bottlenecks[particle_id] = 0.0
                delays[particle_id] = float('inf')
                loss_rates[particle_id] = 1.0
                reliabilities[particle_id] = 0.0
                return
            
            # エッジを検索（CSR形式から）
            edge_found = False
            start_idx = row_ptr[u]
            end_idx = row_ptr[u + 1]
            
            for j in range(start_idx, end_idx):
                if col_idx[j] == v:
                    # エッジが見つかった
                    weight = edge_weights[j]
                    delay = edge_delays[j]
                    loss_log = edge_loss_log_costs[j]
                    rel_cost = edge_reliability_costs[j]
                    
                    if weight < min_bottleneck:
                        min_bottleneck = weight
                    total_delay += delay
                    total_loss_log_cost += loss_log
                    total_reliability_cost += rel_cost
                    edge_found = True
                    break
            
            if not edge_found:
                # エッジが見つからない
                bottlenecks[particle_id] = 0.0
                delays[particle_id] = float('inf')
                loss_rates[particle_id] = 1.0
                reliabilities[particle_id] = 0.0
                return
        
        # 結果を計算
        if min_bottleneck == float('inf'):
            bottlenecks[particle_id] = 0.0
        else:
            bottlenecks[particle_id] = min_bottleneck
        
        delays[particle_id] = total_delay
        
        # loss_rate = 1 - exp(-loss_log_cost)
        # Numba CUDAはmath.expを自動的にlibdevice.expに変換
        if total_loss_log_cost < 100.0:  # オーバーフロー防止
            loss_rates[particle_id] = 1.0 - math.exp(-total_loss_log_cost)
        else:
            loss_rates[particle_id] = 1.0
        
        # reliability = exp(-reliability_cost)
        if total_reliability_cost < 100.0:  # オーバーフロー防止
            reliabilities[particle_id] = math.exp(-total_reliability_cost)
        else:
            reliabilities[particle_id] = 0.0

    # PathEncode + 属性計算 + フィットネス計算統合CUDAカーネル（超最適化版）
    @cuda.jit
    def path_encode_attributes_fitness_kernel(particles, row_ptr, col_idx, src, dst, num_nodes,
                                              limit_len, max_path_len,
                                              edge_weights, edge_delays, edge_loss_log_costs, 
                                              edge_reliability_costs,
                                              max_delay, max_loss, min_rel, P_d, P_l, P_r,
                                              bottlenecks, delays, loss_rates, reliabilities,
                                              fitnesses, is_feasible_flags, path_lengths):
        """
        PathEncode + 属性計算 + フィットネス計算を1つのカーネルに統合（超最適化版）
        - 経路探索中にエッジIDを保存して属性計算を高速化
        - 早期終了による制約違反の早期検出
        - 分岐削減とメモリアクセス最適化
        """
        particle_id = cuda.grid(1)
        num_particles = particles.shape[0]
        
        if particle_id >= num_particles:
            return
        
        # ビットマスク配列で訪問済みノードを管理（O(1)チェック）
        # num_nodesに基づいて動的にサイズを計算（最大2048ノードまで対応）
        bits_array_size = (num_nodes + 31) // 32  # 切り上げ
        if bits_array_size > 64:  # 64 * 32 = 2048ノードまで
            bits_array_size = 64
        visited_bits = cuda.local.array(64, dtype=np.int32)  # 最大2048ノードまで
        for i in range(bits_array_size):
            visited_bits[i] = 0
        
        # ローカル配列で経路とエッジIDを管理
        path = cuda.local.array(512, dtype=np.int32)  # 最大経路長
        edge_ids = cuda.local.array(511, dtype=np.int32)  # エッジID（経路長-1）
        
        # PathEncode: 経路探索（エッジIDを同時に保存）
        current_node = src
        path[0] = src
        path_len = 1
        # srcを訪問済みにマーク
        src_idx = src // 32
        src_bit = src % 32
        if src_idx < bits_array_size:
            visited_bits[src_idx] |= (1 << src_bit)
        
        # 属性計算用の変数を初期化（早期終了のため）
        min_bottleneck = float('inf')
        total_delay = 0.0
        total_loss_log_cost = 0.0
        total_reliability_cost = 0.0
        
        # 経路探索
        while current_node != dst and path_len < max_path_len:
            start_idx = row_ptr[current_node]
            end_idx = row_ptr[current_node + 1]
            
            best_neighbor = -1
            best_edge_id = -1
            highest_prio = -1.0
            
            # 隣接ノードを走査（エッジIDも同時に保存）
            for i in range(start_idx, end_idx):
                neighbor = col_idx[i]
                
                # 訪問済みチェック（O(1)ビットマスク操作）
                neighbor_idx = neighbor // 32
                neighbor_bit = neighbor % 32
                if neighbor_idx >= bits_array_size:
                    continue  # 範囲外のノードはスキップ（通常は発生しない）
                if (visited_bits[neighbor_idx] & (1 << neighbor_bit)) != 0:
                    continue  # 訪問済み
                
                # 優先度チェック（multiprocessing版と完全一致）
                # CPU版: for neighbor in neighbors: if neighbor < limit_len: prio = Particle[neighbor]
                # limit_lenは粒子の次元数（通常はnum_nodesと同じ）
                if neighbor < limit_len:
                    prio = particles[particle_id, neighbor]
                    # CPU版と一致: prio > highest_prio の比較（初期値は-1.0）
                    if prio > highest_prio:
                        highest_prio = prio
                        best_neighbor = neighbor
                        best_edge_id = i  # エッジIDを保存
            
            if best_neighbor == -1:
                # 経路が見つからない（隣接ノードがすべて訪問済み、またはlimit_len未満のノードがない）
                # multiprocessing版と一致: 経路が見つからない場合はfitness=-1.0を返す
                bottlenecks[particle_id] = 0.0
                delays[particle_id] = float('inf')
                loss_rates[particle_id] = 1.0
                reliabilities[particle_id] = 0.0
                fitnesses[particle_id] = -1.0
                is_feasible_flags[particle_id] = 0
                path_lengths[particle_id] = path_len
                return
            
            # エッジ属性を取得（経路探索中に同時に計算）
            weight = edge_weights[best_edge_id]
            delay = edge_delays[best_edge_id]
            loss_log = edge_loss_log_costs[best_edge_id]
            rel_cost = edge_reliability_costs[best_edge_id]
            
            # エッジIDを保存
            edge_ids[path_len - 1] = best_edge_id
            
            # 属性を累積
            if weight < min_bottleneck:
                min_bottleneck = weight
            total_delay += delay
            total_loss_log_cost += loss_log
            total_reliability_cost += rel_cost
            
            # 早期終了チェックは削除（multiprocessing版と一致させるため）
            
            # 次のノードに移動
            current_node = best_neighbor
            path[path_len] = current_node
            path_len += 1
            # 訪問済みにマーク
            best_idx = best_neighbor // 32
            best_bit = best_neighbor % 32
            if best_idx < bits_array_size:
                visited_bits[best_idx] |= (1 << best_bit)
        
        # 経路が見つからなかった場合
        if current_node != dst:
            bottlenecks[particle_id] = 0.0
            delays[particle_id] = float('inf')
            loss_rates[particle_id] = 1.0
            reliabilities[particle_id] = 0.0
            fitnesses[particle_id] = -1.0
            is_feasible_flags[particle_id] = 0
            path_lengths[particle_id] = path_len
            return
        
        # 属性計算は既に完了している（経路探索中に計算済み）
        if path_len < 2:
            bottlenecks[particle_id] = 0.0
            delays[particle_id] = float('inf')
            loss_rates[particle_id] = 1.0
            reliabilities[particle_id] = 0.0
            fitnesses[particle_id] = -1.0
            is_feasible_flags[particle_id] = 0
            path_lengths[particle_id] = path_len
            return
        
        # 結果を計算
        if min_bottleneck == float('inf'):
            bottlenecks[particle_id] = 0.0
        else:
            bottlenecks[particle_id] = min_bottleneck
        
        delays[particle_id] = total_delay
        
        # loss_rate = 1 - exp(-loss_log_cost)
        if total_loss_log_cost < 100.0:
            loss_rates[particle_id] = 1.0 - math.exp(-total_loss_log_cost)
        else:
            loss_rates[particle_id] = 1.0
        
        # reliability = exp(-reliability_cost)
        if total_reliability_cost < 100.0:
            reliabilities[particle_id] = math.exp(-total_reliability_cost)
        else:
            reliabilities[particle_id] = 0.0
        
        # フィットネス計算
        d = total_delay
        l = loss_rates[particle_id]
        r = reliabilities[particle_id]
        
        # ペナルティ計算
        penalty = 0.0
        if d > max_delay:
            penalty += P_d * (d - max_delay)
        if l > max_loss:
            penalty += P_l * (l - max_loss)
        if r < min_rel:
            penalty += P_r * (min_rel - r)
        
        # フィットネス計算
        fitnesses[particle_id] = min_bottleneck - penalty
        
        # 制約充足チェック
        if d <= max_delay and l <= max_loss and r >= min_rel:
            is_feasible_flags[particle_id] = 1
        else:
            is_feasible_flags[particle_id] = 0
        
        path_lengths[particle_id] = path_len

    # 属性計算+フィットネス計算統合CUDAカーネル（後方互換性のため残す）
    @cuda.jit
    def calculate_attributes_and_fitness_kernel(paths, path_lengths, is_valid, row_ptr, col_idx,
                                               edge_weights, edge_delays, edge_loss_log_costs, 
                                               edge_reliability_costs, num_nodes,
                                               max_delay, max_loss, min_rel, P_d, P_l, P_r,
                                               bottlenecks, delays, loss_rates, reliabilities,
                                               fitnesses, is_feasible_flags):
        """
        経路属性とフィットネス値を統合して並列計算（カーネル起動回数を削減）
        """
        particle_id = cuda.grid(1)
        num_particles = paths.shape[0]
        
        if particle_id >= num_particles:
            return
        
        if is_valid[particle_id] == 0:
            bottlenecks[particle_id] = 0.0
            delays[particle_id] = float('inf')
            loss_rates[particle_id] = 1.0
            reliabilities[particle_id] = 0.0
            fitnesses[particle_id] = -1.0
            is_feasible_flags[particle_id] = 0
            return
        
        path_len = path_lengths[particle_id]
        if path_len < 2:
            bottlenecks[particle_id] = 0.0
            delays[particle_id] = float('inf')
            loss_rates[particle_id] = 1.0
            reliabilities[particle_id] = 0.0
            fitnesses[particle_id] = -1.0
            is_feasible_flags[particle_id] = 0
            return
        
        # エッジ属性を集約
        min_bottleneck = float('inf')
        total_delay = 0.0
        total_loss_log_cost = 0.0
        total_reliability_cost = 0.0
        
        for i in range(path_len - 1):
            u = paths[particle_id, i]
            v = paths[particle_id, i + 1]
            
            if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
                bottlenecks[particle_id] = 0.0
                delays[particle_id] = float('inf')
                loss_rates[particle_id] = 1.0
                reliabilities[particle_id] = 0.0
                fitnesses[particle_id] = -1.0
                is_feasible_flags[particle_id] = 0
                return
            
            # エッジを検索（CSR形式から）
            edge_found = False
            start_idx = row_ptr[u]
            end_idx = row_ptr[u + 1]
            
            for j in range(start_idx, end_idx):
                if col_idx[j] == v:
                    # エッジが見つかった
                    weight = edge_weights[j]
                    delay = edge_delays[j]
                    loss_log = edge_loss_log_costs[j]
                    rel_cost = edge_reliability_costs[j]
                    
                    if weight < min_bottleneck:
                        min_bottleneck = weight
                    total_delay += delay
                    total_loss_log_cost += loss_log
                    total_reliability_cost += rel_cost
                    edge_found = True
                    break
            
            if not edge_found:
                # エッジが見つからない
                bottlenecks[particle_id] = 0.0
                delays[particle_id] = float('inf')
                loss_rates[particle_id] = 1.0
                reliabilities[particle_id] = 0.0
                fitnesses[particle_id] = -1.0
                is_feasible_flags[particle_id] = 0
                return
        
        # 結果を計算
        if min_bottleneck == float('inf'):
            bottlenecks[particle_id] = 0.0
        else:
            bottlenecks[particle_id] = min_bottleneck
        
        delays[particle_id] = total_delay
        
        # loss_rate = 1 - exp(-loss_log_cost)
        if total_loss_log_cost < 100.0:  # オーバーフロー防止
            loss_rates[particle_id] = 1.0 - math.exp(-total_loss_log_cost)
        else:
            loss_rates[particle_id] = 1.0
        
        # reliability = exp(-reliability_cost)
        if total_reliability_cost < 100.0:  # オーバーフロー防止
            reliabilities[particle_id] = math.exp(-total_reliability_cost)
        else:
            reliabilities[particle_id] = 0.0
        
        # フィットネス計算（統合）
        d = total_delay
        l = loss_rates[particle_id]
        r = reliabilities[particle_id]
        
        # ペナルティ計算
        penalty = 0.0
        if d > max_delay:
            penalty += P_d * (d - max_delay)
        if l > max_loss:
            penalty += P_l * (l - max_loss)
        if r < min_rel:
            penalty += P_r * (min_rel - r)
        
        # フィットネス計算
        fitnesses[particle_id] = min_bottleneck - penalty
        
        # 制約充足チェック
        if d <= max_delay and l <= max_loss and r >= min_rel:
            is_feasible_flags[particle_id] = 1
        else:
            is_feasible_flags[particle_id] = 0

    # pBest更新CUDAカーネル
    @cuda.jit
    def update_pbest_kernel(swarms, current_fitness, pBests, pBests_fitness):
        """
        GPU上でpBestを更新
        """
        particle_id = cuda.grid(1)
        num_particles = swarms.shape[0]
        
        if particle_id >= num_particles:
            return
        
        # pBest更新チェック
        if current_fitness[particle_id] > pBests_fitness[particle_id]:
            # pBestを更新
            dim = swarms.shape[1]
            for i in range(dim):
                pBests[particle_id, i] = swarms[particle_id, i]
            pBests_fitness[particle_id] = current_fitness[particle_id]

    # PSO更新（速度・位置更新）CUDAカーネル
    @cuda.jit
    def update_velocity_position_kernel(swarms, velocities, pBests, lBests, w, c1, c2, r1, r2):
        """
        GPU上で速度と位置を更新
        multiprocessing版と一致させるため、各粒子に対して1つの乱数値を使用
        """
        particle_id = cuda.grid(1)
        num_particles = swarms.shape[0]
        
        if particle_id >= num_particles:
            return
        
        dim = swarms.shape[1]
        # multiprocessing版と一致: 各粒子に対して1つの乱数値を使用（全次元で同じ値）
        r1_val = r1[particle_id, 0]  # 最初の次元の値を使用（全次元で同じ）
        r2_val = r2[particle_id, 0]  # 最初の次元の値を使用（全次元で同じ）
        
        for i in range(dim):
            velocities[particle_id, i] = (w * velocities[particle_id, i] + 
                                        c1 * r1_val * (pBests[particle_id, i] - swarms[particle_id, i]) + 
                                        c2 * r2_val * (lBests[particle_id, i] - swarms[particle_id, i]))
            swarms[particle_id, i] += velocities[particle_id, i]

    # 【削除】未使用の空間的近傍計算CUDAカーネル
    # このカーネルは実際にはグローバル最良を使用しており、Spatial PSOの実装ではない
    # また、全く使用されていないため削除
    # CuPy版のget_spatial_lbest_gpu()を使用することで、GPU上で完結した実装が可能

# ==========================================
# 4. CPU側フォールバック関数（GPU未検出時用）
# ==========================================

def PathEncode(Particle, Graph, src, dst):
    """CPU側フォールバック用PathEncode"""
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
        
    return path, True

def calculate_path_attributes_4d(G, path):
    """CPU側フォールバック用属性計算"""
    if not path or len(path) < 2: return 0, float('inf'), 1.0, 0.0
    bottleneck = float('inf')
    total_delay = 0
    total_loss_log_cost = 0
    total_reliability_cost = 0
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if not G.has_edge(u, v): return 0, float('inf'), 1.0, 0.0
        edge_data = G.edges[u, v]
        
        bottleneck = min(bottleneck, edge_data.get('weight', 0))
        total_delay += edge_data.get('delay', float('inf'))
        total_loss_log_cost += edge_data.get('loss_log_cost', float('inf'))
        total_reliability_cost += edge_data.get('reliability_cost', float('inf'))
        
    total_loss_rate = 1 - math.exp(-total_loss_log_cost)
    total_reliability = math.exp(-total_reliability_cost)
    return bottleneck, total_delay, total_loss_rate, total_reliability

# ==========================================
# 5. 近傍計算関数（CPU版とGPU版）
# ==========================================

def get_spatial_lbest(swarms, pBests, pBests_fitness, k=5):
    """
    CPU側で空間的近傍を計算（multiprocessing版と同じ実装）
    """
    sq_norms = np.sum(swarms**2, axis=1)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * np.dot(swarms, swarms.T)
    dist_sq = np.maximum(dist_sq, 0)
    nearest_indices = np.argpartition(dist_sq, kth=k, axis=1)[:, :k]
    
    lBest_matrix = np.zeros_like(swarms)
    for i in range(len(swarms)):
        neighbors = nearest_indices[i]
        best_local_idx = np.argmax(pBests_fitness[neighbors])
        lBest_matrix[i] = pBests[neighbors[best_local_idx]]
    return lBest_matrix

# ==========================================
# 6. GPU用近傍計算関数（CuPy）
# ==========================================

def get_spatial_lbest_gpu(swarms_gpu, pBests_gpu, pBests_fitness_gpu, k=5):
    """
    GPU上で空間的近傍を計算（CuPyで完全にGPU上で完結、CPU転送なし）
    【修正】CPU側のforループを削除し、完全にベクトル化されたGPU処理に変更
    """
    # 距離の二乗を計算（GPU上で実行）
    sq_norms = cp.sum(swarms_gpu**2, axis=1)
    # ブロードキャストを使用した効率的な距離計算
    dist_sq = sq_norms[:, cp.newaxis] + sq_norms[cp.newaxis, :] - 2 * cp.dot(swarms_gpu, swarms_gpu.T)
    dist_sq = cp.maximum(dist_sq, 0)  # 数値安定性のため
    
    # k個の最近傍を取得（GPU上で実行）
    nearest_indices = cp.argpartition(dist_sq, kth=k, axis=1)[:, :k]  # (num_particles, k)
    
    # lBestを計算（完全にGPU上で実行、ベクトル化）
    num_particles = swarms_gpu.shape[0]
    
    # 【修正】ベクトル化された処理: 全粒子を並列処理
    # 各粒子について、近傍k個のfitnessを取得
    # nearest_indices: (num_particles, k)
    # pBests_fitness_gpu[nearest_indices]: (num_particles, k) - 各粒子の近傍k個のfitness
    neighbor_fitness_all = pBests_fitness_gpu[nearest_indices]  # (num_particles, k)
    
    # 各粒子について、近傍k個の中で最良のインデックスを見つける
    best_local_indices = cp.argmax(neighbor_fitness_all, axis=1)  # (num_particles,)
    
    # 最良の近傍粒子のインデックスを取得
    # nearest_indices[i, best_local_indices[i]] を全粒子について取得
    particle_indices = cp.arange(num_particles)  # (num_particles,)
    best_neighbor_indices = nearest_indices[particle_indices, best_local_indices]  # (num_particles,)
    
    # lBestを取得（ベクトル化）
    lBest_matrix = pBests_gpu[best_neighbor_indices]  # (num_particles, dim)
    
    return lBest_matrix

def get_global_lbest_gpu(swarms_gpu, gBest_gpu):
    """
    GPU上でグローバルlBestを計算
    """
    return cp.tile(gBest_gpu, (swarms_gpu.shape[0], 1))

# ==========================================
# 7. 厳密解法（変更なし）
# ==========================================

def find_optimal_path_by_label_correcting_4d(G, source, target, max_delay, max_loss_rate, min_reliability):
    # 厳密解法は並列化が難しいため、シングルスレッド実行
    labels = {node: [] for node in G.nodes()}
    pq = []
    initial_label = (0.0, float('inf'), 0.0, 0.0) # (delay, neg_bn, loss, rel_cost)
    labels[source].append(initial_label)
    
    heappush(pq, (-initial_label[1], initial_label[0], initial_label[2], initial_label[3], source, initial_label))
    
    min_reliability_cost = -math.log(min_reliability) if min_reliability > 0 else float('inf')
    max_loss_log_cost = -math.log(1 - max_loss_rate) if max_loss_rate < 1 else float('inf')

    best_bottle = -1

    while pq:
        neg_bottle, d_curr, l_curr, r_curr, u, label_curr = heappop(pq)
        
        # Label cleaning check
        if label_curr not in labels[u]: continue
        
        for v in G.neighbors(u):
            edge = G.get_edge_data(u, v)
            d_new = d_curr + edge.get("delay", 0)
            b_new = min(-neg_bottle, edge.get("weight", 1))
            l_new = l_curr + edge.get("loss_log_cost", 0)
            r_new = r_curr + edge.get("reliability_cost", 0)
            
            if d_new > max_delay or l_new > max_loss_log_cost or r_new > min_reliability_cost: continue
            
            label_new = (d_new, b_new, l_new, r_new)
            
            # Dominance check
            is_dominated = False
            for d, b, l, r in labels[v]:
                # Dominance: new label is worse if Delay>=, BN<=, Loss>=, RelCost>=
                if d <= d_new and b >= b_new and l <= l_new and r <= r_new:
                    is_dominated = True
                    break
            if is_dominated: continue
            
            # Remove dominated labels
            labels[v] = [lbl for lbl in labels[v] if not (lbl[0] >= d_new and lbl[1] <= b_new and lbl[2] >= l_new and lbl[3] >= r_new)]
            labels[v].append(label_new)
            
            # Sort order priority: Maximize BN -> Minimize NegBN
            heappush(pq, (-b_new, d_new, l_new, r_new, v, label_new))

    final_labels = labels.get(target, [])
    if not final_labels: return -1
    
    for d, b, l, r in final_labels:
        if b > best_bottle: best_bottle = b
    return best_bottle

# ==========================================
# 8. Numba CUDA + CuPy並列化 PSO シミュレーション関数
# ==========================================

def run_pso_cuda(Graph, src, dst, constraints, pso_params, topology='spatial', enable_restart=True):
    """
    Numba CUDA + CuPyを用いた並列PSO
    - 粒子の評価（PathEncode + 属性計算）: GPU側でNumba CUDAカーネル実行
    - PSO更新処理（位置更新、速度更新、近傍計算）: GPU側でCuPy実行
    """
    # GPU検出（Numba CUDAとCuPyを分離）
    use_numba_cuda = init_gpu() and NUMBA_CUDA_AVAILABLE
    use_cupy = CUDA_AVAILABLE and use_numba_cuda
    use_gpu = use_numba_cuda  # Numba CUDAがあれば評価はGPUで実行
    
    # CuPy使用状況を表示（デバッグ情報付き）
    if use_cupy:
        print(f"[INFO] CuPy enabled: Using GPU for PSO updates (spatial topology, velocity updates)")
    elif CUDA_AVAILABLE and not use_numba_cuda:
        print(f"[INFO] CuPy available but Numba CUDA not available. Using CPU for evaluation.")
        print(f"[DEBUG] CUDA_AVAILABLE={CUDA_AVAILABLE}, use_numba_cuda={use_numba_cuda}, NUMBA_CUDA_AVAILABLE={NUMBA_CUDA_AVAILABLE}")
    elif not CUDA_AVAILABLE:
        print(f"[INFO] CuPy not available. Using CPU for all operations.")
        print(f"[DEBUG] CUDA_AVAILABLE={CUDA_AVAILABLE}, use_numba_cuda={use_numba_cuda}, NUMBA_CUDA_AVAILABLE={NUMBA_CUDA_AVAILABLE}")
    
    # 事前経路計算 (遅延制約の上限設定用)
    try:
        min_delay_path = nx.dijkstra_path(Graph, source=src, target=dst, weight='delay')
        _, min_delay, _, _ = calculate_path_attributes_4d(Graph, min_delay_path)
        abs_max_delay = min_delay * constraints['delay_multiplier']
        constraints_for_eval = constraints.copy()
        constraints_for_eval['delay_multiplier'] = abs_max_delay
    except:
        return -1.0, 0.0

    num_nodes = len(Graph.nodes())
    num_par = pso_params['num_par']
    num_gen = pso_params['num_gen']
    
    # パラメータ展開
    w_start, w_end = pso_params['w_config']
    c1_start, c1_end = pso_params['c1_config']
    c2_start, c2_end = pso_params['c2_config']
    Pd_start, Pd_end = pso_params['Pd_config']
    Pl_start, Pl_end = pso_params['Pl_config']
    Pr_start, Pr_end = pso_params['Pr_config']
    
    TIME_LIMIT_SEC = pso_params['time_limit_sec']
    pso_start_time = time.time()
    
    # Restart関連
    RESTART_THRESHOLD = 20
    restart_counter = 0
    
    # グラフを隣接リスト形式に変換（Numba CUDA使用時のみ）
    if use_numba_cuda:
        t_graph_start = time.time()
        adj_list, adj_ptr, edge_weights, edge_delays, edge_loss_log_costs, edge_reliability_costs = graph_to_adjacency_list(Graph)
        t_graph_conv = time.time()
        
        # GPUメモリに転送
        d_row_ptr = cuda.to_device(adj_ptr)  # 隣接リストのポインタ（CSR形式と同じ構造）
        d_col_idx = cuda.to_device(adj_list)  # 隣接ノードリスト（CSR形式と同じ構造）
        d_edge_weights = cuda.to_device(edge_weights)
        d_edge_delays = cuda.to_device(edge_delays)
        d_edge_loss_log_costs = cuda.to_device(edge_loss_log_costs)
        d_edge_reliability_costs = cuda.to_device(edge_reliability_costs)
        t_gpu_transfer = time.time()
        print(f"[PROFILE] GPU転送: {t_gpu_transfer - t_graph_conv:.3f}s")
    
    # 粒子初期化（元コードと完全一致: float64で初期化）
    # 重要: 元コードと同じ乱数シードを使用するため、同じ初期化方法を維持
    swarms_cpu = np.random.uniform(1, 20, (num_par, num_nodes)).astype(np.float64)
    
    # Numba CUDA用のGPU配列（PSO更新処理用）
    # 注意: GPU計算ではfloat32を使用するが、CPU側ではfloat64を維持
    use_numba_cuda_updates = use_numba_cuda  # Numba CUDAでPSO更新も実行
    if use_numba_cuda_updates:
        # GPU計算用にfloat32に変換（評価時のみ）
        swarms_gpu = cuda.to_device(swarms_cpu.astype(np.float32))
        velocities_gpu = cuda.to_device(np.zeros((num_par, num_nodes), dtype=np.float32))
        pBests_gpu = cuda.to_device(swarms_cpu.copy().astype(np.float32))
        pBests_fitness_gpu = cuda.to_device(np.full(num_par, -float('inf'), dtype=np.float32))
        lBests_gpu = cuda.device_array((num_par, num_nodes), dtype=np.float32)
        r1_gpu = cuda.device_array((num_par, 1), dtype=np.float32)  # multiprocessing版と一致: (num_par, 1)
        r2_gpu = cuda.device_array((num_par, 1), dtype=np.float32)  # multiprocessing版と一致: (num_par, 1)
        print("Numba CUDA arrays initialized. Using GPU for PSO updates.")
    
    # CuPyが使用可能な場合はGPUメモリに配置、そうでなければCPU
    if use_cupy:
        try:
            # CuPyが実際に動作するかテスト（CUDA 11対応）
            test_gpu = cp.asarray(np.array([1.0], dtype=np.float32))
            _ = test_gpu + 1.0
            # より大きな配列でテスト（メモリ確保の確認）
            test_large = cp.zeros((100, 100), dtype=cp.float32)
            _ = test_large + 1.0
            del test_gpu, test_large
            
            # GPUメモリに転送（float32で効率化、メモリ効率を考慮）
            # 重要: CuPy配列はGPU上で直接作成し、CPU転送を最小限に
            swarms_gpu_cupy = cp.asarray(swarms_cpu.astype(np.float32))
            velocities_gpu_cupy = cp.zeros((num_par, num_nodes), dtype=cp.float32)
            pBests_gpu_cupy = cp.array(swarms_gpu_cupy, copy=True)  # GPU上でコピー
            pBests_fitness_gpu_cupy = cp.full(num_par, -float('inf'), dtype=cp.float32)
            
            # CuPyのバージョン情報とGPU情報を表示
            try:
                cupy_version = cp.__version__
                device_info = cp.cuda.Device()
                meminfo = cp.cuda.runtime.memGetInfo()
                print(f"CuPy initialized successfully (version: {cupy_version}).")
                print(f"  GPU Device: {device_info.id}, Memory: {meminfo[0] / 1024**3:.2f} GB free / {meminfo[1] / 1024**3:.2f} GB total")
                print(f"  Using GPU for PSO updates (spatial topology, velocity updates).")
            except:
                print("CuPy initialized successfully. Using GPU for PSO updates.")
        except Exception as e:
            print(f"Warning: CuPy operation failed: {e}")
            print("Falling back to CPU for PSO updates.")
            print("Note: For CUDA 11.x, ensure CuPy is installed with: pip install cupy-cuda11x")
            use_cupy = False
    
    # CPU側の配列（常に必要、元コードと完全一致: float64）
    # 注意: swarms_cpuは既にfloat64で初期化済み
    velocities_cpu = np.zeros_like(swarms_cpu, dtype=np.float64)
    pBests_cpu = np.array(swarms_cpu, dtype=np.float64)
    pBests_fitness_cpu = np.full(num_par, -float('inf'), dtype=np.float64)
    
    gBest_cpu = swarms_cpu[0].copy()
    gBest_gpu = None
    gBest_gpu_cupy = None
    if use_numba_cuda_updates:
        gBest_gpu = cuda.to_device(swarms_cpu[0].astype(np.float32))
    if use_cupy:
        try:
            # float32で効率化
            gBest_gpu_cupy = cp.asarray(gBest_cpu.astype(np.float32))
        except Exception as e:
            print(f"Warning: gBest_gpu_cupy initialization failed: {e}")
            use_cupy = False
    gBest_fitness = -float('inf')
    gBest_feasible_bn = -1
    last_best_bn = -1.0
    
    # 最大経路長の設定（現実的な上限を設定）
    # 通常、最短経路はnum_nodesよりはるかに短いが、安全のためnum_nodesの2倍を上限とする
    MAX_PATH_LEN = min(512, num_nodes * 2)  # メモリ制約と現実的な経路長を考慮
    
    # Numba CUDA評価用の出力配列（事前割り当て）
    if use_numba_cuda:
        paths_gpu = cuda.device_array((num_par, MAX_PATH_LEN), dtype=np.int32)
        path_lengths_gpu = cuda.device_array(num_par, dtype=np.int32)
        is_valid_gpu = cuda.device_array(num_par, dtype=np.int32)
        bottlenecks_gpu = cuda.device_array(num_par, dtype=np.float32)
        delays_gpu = cuda.device_array(num_par, dtype=np.float32)
        loss_rates_gpu = cuda.device_array(num_par, dtype=np.float32)
        reliabilities_gpu = cuda.device_array(num_par, dtype=np.float32)
        fitnesses_gpu = cuda.device_array(num_par, dtype=np.float32)
        is_feasible_flags_gpu = cuda.device_array(num_par, dtype=np.int32)
        
        # CuPyが使用可能な場合、評価結果をCuPy配列としても保持（GPU上で完結させるため）
        fitnesses_gpu_cupy = None
        is_feasible_flags_gpu_cupy = None
        bottlenecks_gpu_cupy = None
        if use_cupy:
            try:
                fitnesses_gpu_cupy = cp.zeros(num_par, dtype=cp.float32)
                is_feasible_flags_gpu_cupy = cp.zeros(num_par, dtype=cp.int32)
                bottlenecks_gpu_cupy = cp.zeros(num_par, dtype=cp.float32)
            except:
                pass  # CuPy配列の作成に失敗した場合は従来通り
        
        # スレッドブロック設定（最適化：GPU利用率を向上）
        # 100粒子に対して、32スレッド/ブロック → 4ブロック、64スレッド/ブロック → 2ブロック
        threads_per_block = 32  # より小さいブロックサイズで複数ブロックを確保
        blocks_per_grid = (num_par + threads_per_block - 1) // threads_per_block
        print(f"[PROFILE] スレッドブロック設定: {blocks_per_grid}ブロック x {threads_per_block}スレッド/ブロック")

    for i in range(num_gen):
        if time.time() - pso_start_time > TIME_LIMIT_SEC: break
        
        t_gen_start = time.time()

        # --- Restart Logic ---
        if enable_restart:
            if gBest_feasible_bn > last_best_bn:
                restart_counter = 0
                last_best_bn = gBest_feasible_bn
            else:
                restart_counter += 1
            
                if restart_counter >= RESTART_THRESHOLD:
                    # 【デバッグ】Restart発生をログに出力
                    print(f"[RESTART] Gen {i+1}: Restart発生 (counter={restart_counter}, gBest_feasible_bn={gBest_feasible_bn:.3f}, last_best_bn={last_best_bn:.3f})")
                    
                    # Explosion（元コードと完全一致）
                    swarms_cpu_new = np.random.uniform(1, 20, (num_par, num_nodes)).astype(np.float64)
                    # 重要: Elite保持を最初に行う（元コード: swarms[0] = gBest）
                    if gBest_cpu is not None:
                        swarms_cpu_new[0] = gBest_cpu.copy()
                        print(f"[RESTART] Elite保持: gBestをswarms[0]に設定")
                    
                    swarms_cpu = swarms_cpu_new
                    velocities_cpu = np.zeros_like(swarms_cpu, dtype=np.float64)
                    pBests_cpu = np.array(swarms_cpu, dtype=np.float64)
                    pBests_fitness_cpu = np.full(num_par, -float('inf'), dtype=np.float64)
                    
                    if use_numba_cuda_updates:
                        # GPU配列を更新（CPU側のswarms_cpuから生成）
                        swarms_gpu = cuda.to_device(swarms_cpu.astype(np.float32))
                        velocities_gpu = cuda.to_device(np.zeros((num_par, num_nodes), dtype=np.float32))
                        pBests_gpu = cuda.to_device(swarms_cpu.astype(np.float32))
                        pBests_fitness_gpu = cuda.to_device(np.full(num_par, -float('inf'), dtype=np.float32))
                        # gBest_gpuも更新（次回使用するため）
                        if gBest_cpu is not None:
                            gBest_gpu = cuda.to_device(gBest_cpu.astype(np.float32))
                    
                    if use_cupy:
                        try:
                            # CPU側のswarms_cpuから生成（Elite保持済み、float32で効率化）
                            swarms_gpu_cupy = cp.asarray(swarms_cpu.astype(np.float32))
                            velocities_gpu_cupy = cp.zeros((num_par, num_nodes), dtype=cp.float32)
                            pBests_gpu_cupy = cp.array(swarms_gpu_cupy, copy=True)  # GPU上でコピー
                            pBests_fitness_gpu_cupy = cp.full(num_par, -float('inf'), dtype=cp.float32)
                            # gBest_gpu_cupyも更新
                            if gBest_cpu is not None:
                                gBest_gpu_cupy = cp.asarray(gBest_cpu.astype(np.float32))
                        except Exception as e:
                            print(f"Warning: CuPy restart failed: {e}")
                            use_cupy = False
                    restart_counter = 0

        # --- Update Parameters ---
        progress = i / num_gen
        w = w_start - (w_start - w_end) * progress
        c1 = c1_start - (c1_start - c1_end) * progress
        c2 = c2_start + (c2_end - c2_start) * progress
        P_d = Pd_start + (Pd_end - Pd_start) * progress
        P_l = Pl_start + (Pl_end - Pl_start) * progress
        P_r = Pr_start + (Pr_end - Pr_start) * progress
        
        # --- Numba CUDA側並列評価 (Evaluation) ---
        if use_numba_cuda:
            t_eval_start = time.time()
            # 粒子位置をGPUメモリに転送（Numba CUDA用）
            if use_numba_cuda_updates:
                # Numba CUDA配列を直接使用（GPU上で完結）
                # 型を確認: float32である必要がある
                d_particles = swarms_gpu
            elif use_cupy:
                # CuPy配列からNumba CUDA配列に変換（GPU内転送）
                d_particles = cuda.to_device(swarms_gpu_cupy.get().astype(np.float32))
            else:
                # CPU配列からNumba CUDA配列に転送
                d_particles = cuda.to_device(swarms_cpu.astype(np.float32))
            t_particle_transfer = time.time()
            
            # PathEncode + 属性計算 + フィットネス計算統合カーネル実行（最適化版）
            t_eval_kernel_start = time.time()
            # limit_lenは粒子の次元数（multiprocessing版: len(Particle) = num_nodes）
            limit_len_val = d_particles.shape[1]  # 粒子の次元数
            path_encode_attributes_fitness_kernel[blocks_per_grid, threads_per_block](
                d_particles, d_row_ptr, d_col_idx,
                np.int32(src), np.int32(dst), np.int32(num_nodes),
                np.int32(limit_len_val),  # limit_len: 粒子の次元数（multiprocessing版と一致）
                np.int32(MAX_PATH_LEN),
                d_edge_weights, d_edge_delays, d_edge_loss_log_costs, d_edge_reliability_costs,
                np.float32(abs_max_delay), np.float32(constraints['loss_constraint']),
                np.float32(constraints['reliability_constraint']),
                np.float32(P_d), np.float32(P_l), np.float32(P_r),
                bottlenecks_gpu, delays_gpu, loss_rates_gpu, reliabilities_gpu,
                fitnesses_gpu, is_feasible_flags_gpu, path_lengths_gpu
            )
            cuda.synchronize()  # 結果を使用する前に同期
            t_eval_kernel_end = time.time()
            
            if i < 3 or i % 10 == 0:  # 最初の3世代と10世代ごとに出力
                # デバッグ情報: 経路探索の成功率と実行可能解の数を確認
                is_feasible_debug = is_feasible_flags_gpu.copy_to_host()
                fitnesses_debug = fitnesses_gpu.copy_to_host()
                path_lengths_debug = path_lengths_gpu.copy_to_host()
                bottlenecks_debug = bottlenecks_gpu.copy_to_host()
                delays_debug = delays_gpu.copy_to_host()
                loss_rates_debug = loss_rates_gpu.copy_to_host()
                reliabilities_debug = reliabilities_gpu.copy_to_host()
                
                valid_paths = np.sum(path_lengths_debug > 1)  # 経路が見つかった数（path_len > 1）
                feasible_count = np.sum(is_feasible_debug == 1)
                # フィットネス値が-1でない粒子の数（有効な経路が見つかった数）
                valid_fitness = np.sum(fitnesses_debug > -1.0)
                
                # 【デバッグ追加】実行可能解が見つからない原因を分析
                if feasible_count == 0 and valid_fitness > 0:
                    # 有効な経路はあるが実行可能解がない場合、制約違反の詳細を確認
                    valid_indices = np.where(fitnesses_debug > -1.0)[0]
                    if len(valid_indices) > 0:
                        sample_idx = valid_indices[0]
                        delay_violations = np.sum(delays_debug[valid_indices] > abs_max_delay)
                        loss_violations = np.sum(loss_rates_debug[valid_indices] > constraints['loss_constraint'])
                        rel_violations = np.sum(reliabilities_debug[valid_indices] < constraints['reliability_constraint'])
                        print(f"[DEBUG] Gen {i+1}: 実行可能解なし - "
                              f"有効経路={valid_fitness}, "
                              f"遅延違反={delay_violations}, "
                              f"損失違反={loss_violations}, "
                              f"信頼性違反={rel_violations}, "
                              f"サンプル(D={delays_debug[sample_idx]:.1f}/{abs_max_delay:.1f}, "
                              f"L={loss_rates_debug[sample_idx]:.3f}/{constraints['loss_constraint']:.3f}, "
                              f"R={reliabilities_debug[sample_idx]:.3f}/{constraints['reliability_constraint']:.3f})")
                elif valid_fitness == 0:
                    # 経路が見つからない場合
                    print(f"[DEBUG] Gen {i+1}: 経路が見つからない粒子が100% - PathEncodeの問題の可能性")
                
                print(f"[PROFILE] Gen {i+1} - 評価: "
                      f"転送={t_particle_transfer-t_eval_start:.3f}s "
                      f"統合カーネル={t_eval_kernel_end-t_eval_kernel_start:.3f}s "
                      f"合計={t_eval_kernel_end-t_eval_start:.3f}s "
                      f"有効経路={valid_fitness}/100 実行可能解={feasible_count}/100 "
                      f"gBest_feasible_bn={gBest_feasible_bn:.3f}")
            
            
            # 評価結果の処理（Numba CUDA配列を使用）
            if use_numba_cuda_updates:
                # Numba CUDA配列で処理（GPU上で完結）
                # gBest_feasible_bnの更新（CPU側で実行）
                fitnesses_cpu = fitnesses_gpu.copy_to_host()
                is_feasible_cpu = is_feasible_flags_gpu.copy_to_host()
                bottlenecks_cpu = bottlenecks_gpu.copy_to_host()
                
                for j in range(num_par):
                    if is_feasible_cpu[j] == 1 and bottlenecks_cpu[j] > gBest_feasible_bn:
                        gBest_feasible_bn = bottlenecks_cpu[j]
                
                # GPU上でpBestを更新
                update_pbest_kernel[blocks_per_grid, threads_per_block](
                    d_particles, fitnesses_gpu, pBests_gpu, pBests_fitness_gpu
                )
                cuda.synchronize()
                
                # GPU上でgBestを更新（最良のfitness値を持つ粒子を探す）
                best_idx = int(np.argmax(fitnesses_cpu))
                best_fitness = float(fitnesses_cpu[best_idx])
                if best_fitness > gBest_fitness:
                    gBest_fitness = best_fitness
                    particles_host = d_particles.copy_to_host()
                    gBest_cpu = particles_host[best_idx].astype(np.float64)
                    # GPU側のgBestも更新（次回使用するため）
                    gBest_gpu = cuda.to_device(particles_host[best_idx])
                
                # CPU側の配列も更新（互換性のため）
                # 重要: 評価に使った粒子位置をCPU側にも反映
                current_fitness_cpu = fitnesses_cpu
                current_best_idx = best_idx
                pBests_fitness_cpu = pBests_fitness_gpu.copy_to_host()
                # 評価に使った粒子位置をCPU側に反映（速度更新で使用するため）
                swarms_cpu = d_particles.copy_to_host().astype(np.float64)
                # GPU側のswarms_gpuも更新（速度更新で使用するため）
                swarms_gpu = d_particles  # 既にGPU上にあるので、そのまま使用
            elif use_cupy:
                try:
                    # 【最適化】Numba CUDA配列からCuPy配列への変換を効率化
                    # 一度にすべての配列をCPUに転送してからCuPy配列に変換（メモリ転送をまとめる）
                    t_cupy_transfer_start = time.time()
                    
                    # 評価に使った粒子位置をCuPy配列として取得
                    particles_host = d_particles.copy_to_host().astype(np.float32)
                    swarms_evaluated_gpu = cp.asarray(particles_host)
                    
                    # Numba CUDA配列からCuPy配列にコピー（一度CPU経由、バッチ転送で効率化）
                    fitnesses_host = fitnesses_gpu.copy_to_host().astype(np.float32)
                    is_feasible_host = is_feasible_flags_gpu.copy_to_host().astype(np.int32)
                    bottlenecks_host = bottlenecks_gpu.copy_to_host().astype(np.float32)
                    
                    # GPUメモリに一括転送（CuPy配列として）
                    fitnesses_gpu_cupy = cp.asarray(fitnesses_host)
                    is_feasible_flags_gpu_cupy = cp.asarray(is_feasible_host)
                    bottlenecks_gpu_cupy = cp.asarray(bottlenecks_host)
                    
                    t_cupy_transfer_end = time.time()
                    if i < 3 or i % 10 == 0:
                        print(f"[PROFILE] Gen {i+1} - Numba→CuPy転送: {t_cupy_transfer_end-t_cupy_transfer_start:.3f}s")
                    
                    # GPU上でgBest_feasible_bnを更新（実行可能解の中で最良のbottleneckを探す）
                    feasible_mask = is_feasible_flags_gpu_cupy == 1
                    feasible_count_gpu = int(cp.sum(feasible_mask))
                    if cp.any(feasible_mask):
                        feasible_bottlenecks = bottlenecks_gpu_cupy[feasible_mask]
                        best_feasible_bn_gpu = float(cp.max(feasible_bottlenecks))
                        if best_feasible_bn_gpu > gBest_feasible_bn:
                            gBest_feasible_bn = best_feasible_bn_gpu
                    
                    # GPU上でpBestを更新（CuPyのベクトル化された操作を使用）
                    # 元コードと一致: current_fitness > pBests_fitness の場合に更新
                    update_mask_gpu = fitnesses_gpu_cupy > pBests_fitness_gpu_cupy
                    # マスクを使用してpBestを更新（GPU上で完結、メモリ効率的）
                    if cp.any(update_mask_gpu):
                        pBests_gpu_cupy[update_mask_gpu] = swarms_evaluated_gpu[update_mask_gpu]
                        pBests_fitness_gpu_cupy[update_mask_gpu] = fitnesses_gpu_cupy[update_mask_gpu]
                    
                    # GPU上でgBestを更新（最良のfitness値を持つ粒子を探す）
                    best_idx_gpu = int(cp.argmax(fitnesses_gpu_cupy))
                    best_fitness_gpu = float(fitnesses_gpu_cupy[best_idx_gpu])
                    if best_fitness_gpu > gBest_fitness:
                        gBest_fitness = best_fitness_gpu
                        # 重要: 評価に使った粒子位置からgBestを取得（GPU上でコピー）
                        gBest_gpu_cupy = swarms_evaluated_gpu[best_idx_gpu].copy()
                        # CPU側にも反映（必要時のみ）
                        gBest_cpu = gBest_gpu_cupy.get().astype(np.float64)
                    
                    # CPU側の配列も更新（互換性のため、最小限の転送）
                    # 注意: これらの転送は必要最小限（デバッグや互換性のため）
                    current_fitness_cpu = fitnesses_gpu_cupy.get()
                    current_best_idx = best_idx_gpu
                    # CPU側のpBests_fitnessも更新（互換性のため）
                    pBests_fitness_cpu = pBests_fitness_gpu_cupy.get()
                    # CPU側のswarms_cpuも更新（評価に使った粒子位置を反映）
                    swarms_cpu = swarms_evaluated_gpu.get().astype(np.float64)
                    # swarms_gpu_cupyも更新（次回の速度更新で使用）
                    swarms_gpu_cupy = swarms_evaluated_gpu
                except Exception as e:
                    # CuPy操作が失敗した場合のフォールバック
                    print(f"Warning: CuPy evaluation processing failed: {e}")
                    # 重要: 評価に使った粒子位置もCPU側に反映
                    swarms_cpu = d_particles.copy_to_host().astype(np.float64)
                    
                    current_fitness_cpu = fitnesses_gpu.copy_to_host()
                    is_feasible_cpu = is_feasible_flags_gpu.copy_to_host()
                    bottlenecks_cpu = bottlenecks_gpu.copy_to_host()
                    
                    # gBest (Feasible) の更新
                    for j in range(num_par):
                        if is_feasible_cpu[j] == 1 and bottlenecks_cpu[j] > gBest_feasible_bn:
                            gBest_feasible_bn = bottlenecks_cpu[j]
                    
                    # fitnesses_gpu_cupyをNoneに設定（CPU側で更新処理を行うため）
                    fitnesses_gpu_cupy = None
                    use_cupy = False  # 次回からCuPyを使用しない
            else:
                # CuPyが使用できない場合、従来通りCPUに転送
                t_transfer_start = time.time()
                # 重要: 評価に使った粒子位置もCPU側に反映
                swarms_cpu = d_particles.copy_to_host().astype(np.float64)
                
                current_fitness_cpu = fitnesses_gpu.copy_to_host()
                is_feasible_cpu = is_feasible_flags_gpu.copy_to_host()
                bottlenecks_cpu = bottlenecks_gpu.copy_to_host()
                # 注意: 統合カーネルではis_valid_gpuは存在しない（path_lengths_gpuから推測可能）
                path_lengths_cpu = path_lengths_gpu.copy_to_host()
                is_valid_cpu = (path_lengths_cpu > 1).astype(np.int32)  # 経路が見つかったかどうか
                t_transfer_end = time.time()
                
                if i < 3 or i % 10 == 0:  # 最初の3世代と10世代ごとに出力
                    print(f"[PROFILE] Gen {i+1} - GPU→CPU転送: {t_transfer_end-t_transfer_start:.3f}s")
                
                # gBest (Feasible) の更新
                feasible_count = np.sum(is_feasible_cpu == 1)
                for j in range(num_par):
                    if is_feasible_cpu[j] == 1 and bottlenecks_cpu[j] > gBest_feasible_bn:
                        gBest_feasible_bn = bottlenecks_cpu[j]
                
                # デバッグ: 最初の2世代で簡潔に出力
                if i < 2:
                    valid_path_count = np.sum(is_valid_cpu == 1)
                    print(f"G{i+1}: paths={valid_path_count} feas={feasible_count} BN={gBest_feasible_bn:.1f}", end="")
                    if feasible_count > 0:
                        feasible_bns = bottlenecks_cpu[is_feasible_cpu == 1]
                        print(f" bns={feasible_bns[0]:.1f}")
                    elif valid_path_count > 0:
                        # 有効な経路はあるが実行可能解がない場合、制約違反の詳細を確認
                        valid_indices = np.where(is_valid_cpu == 1)[0]
                        if len(valid_indices) > 0:
                            sample_idx = valid_indices[0]
                            delays_sample = delays_gpu.copy_to_host()
                            loss_rates_sample = loss_rates_gpu.copy_to_host()
                            reliabilities_sample = reliabilities_gpu.copy_to_host()
                            print(f" (D={delays_sample[sample_idx]:.1f}/{abs_max_delay:.1f} L={loss_rates_sample[sample_idx]:.3f} R={reliabilities_sample[sample_idx]:.3f})")
                    else:
                        print()
        
        else:
            # CPU側フォールバック（逐次実行）
            current_fitness_cpu = np.zeros(num_par)
            for j in range(num_par):
                path, is_valid = PathEncode(swarms_cpu[j], Graph, src, dst)
                if not is_valid:
                    current_fitness_cpu[j] = -1.0
                    continue
                
                bn, d, l, r = calculate_path_attributes_4d(Graph, path)
                
                penalty = 0
                if d > abs_max_delay: penalty += P_d * (d - abs_max_delay)
                if l > constraints['loss_constraint']: penalty += P_l * (l - constraints['loss_constraint'])
                if r < constraints['reliability_constraint']: penalty += P_r * (constraints['reliability_constraint'] - r)
                
                current_fitness_cpu[j] = bn - penalty
                is_feasible = (d <= abs_max_delay and l <= constraints['loss_constraint'] and r >= constraints['reliability_constraint'])
                if is_feasible and bn > gBest_feasible_bn:
                    gBest_feasible_bn = bn
                
                # デバッグ: 最初の数世代で実行可能解の数を確認
                if i < 5 and j == 0:
                    feasible_count_cpu = sum(1 for k in range(num_par) 
                                            if current_fitness_cpu[k] != -1.0 and 
                                            (d <= abs_max_delay if k == j else True))  # 簡易チェック
                    if feasible_count_cpu > 0:
                        print(f"Gen {i+1}: CPU evaluation, checking feasibility...")

        # --- pBest / gBest (Fitness based) 更新 ---
        # CuPyが使用可能でGPU上で更新済みの場合は、CPU側も同期
        if not (use_cupy and fitnesses_gpu_cupy is not None):
            # CPU側で更新（CuPyが使用できない場合）
            update_mask = current_fitness_cpu > pBests_fitness_cpu
            pBests_cpu[update_mask] = swarms_cpu[update_mask]
            pBests_fitness_cpu[update_mask] = current_fitness_cpu[update_mask]
            
            current_best_idx = np.argmax(current_fitness_cpu)
            if current_fitness_cpu[current_best_idx] > gBest_fitness:
                gBest_fitness = current_fitness_cpu[current_best_idx]
                gBest_cpu = swarms_cpu[current_best_idx].copy()
            
            # CuPyが使用可能な場合、GPU側にも反映
            if use_cupy:
                try:
                    current_fitness_gpu_cupy = cp.asarray(current_fitness_cpu.astype(np.float32))
                    update_mask_gpu = current_fitness_gpu_cupy > pBests_fitness_gpu_cupy
                    pBests_gpu_cupy[update_mask_gpu] = swarms_gpu_cupy[update_mask_gpu]
                    pBests_fitness_gpu_cupy[update_mask_gpu] = current_fitness_gpu_cupy[update_mask_gpu]
                    # gBest_gpu_cupyを更新
                    gBest_gpu_cupy = cp.asarray(gBest_cpu.astype(np.float32))
                except:
                    use_cupy = False
        else:
            # GPU上で更新済みの場合、CPU側の配列も同期（互換性のため）
            pBests_cpu = pBests_gpu_cupy.get().astype(np.float64)
            pBests_fitness_cpu = pBests_fitness_gpu_cupy.get()
            swarms_cpu = swarms_gpu_cupy.get().astype(np.float64)
        
        # --- Topology Selection & Velocity Update ---
        t_update_start = time.time()
        if use_numba_cuda_updates:
            # Numba CUDAでPSO更新を実行
            # lBestの計算（CPU側で計算してGPUに転送）
            t_lbest_start = time.time()
            if topology == 'global':
                # グローバルトポロジー: すべての粒子がgBestを参照
                if gBest_gpu is not None:
                    # gBestを各粒子にコピー
                    gBest_host = gBest_gpu.copy_to_host()
                    lBests_host = np.tile(gBest_host, (num_par, 1))
                    lBests_gpu = cuda.to_device(lBests_host.astype(np.float32))
                else:
                    # 【修正】gBestがまだない場合は、最初の世代の最良粒子（pBestの中で最良）を使用
                    # 元コードでは、最初の世代では全粒子のpBestの中で最良のものをgBestとして使用
                    pBests_fitness_host = pBests_fitness_gpu.copy_to_host()
                    best_pbest_idx = int(np.argmax(pBests_fitness_host))
                    pBests_host = pBests_gpu.copy_to_host()
                    best_pbest = pBests_host[best_pbest_idx]
                    lBests_host = np.tile(best_pbest, (num_par, 1))
                    lBests_gpu = cuda.to_device(lBests_host.astype(np.float32))
                    # デバッグ: 最初の世代でgBestが未初期化の場合
                    if i == 0:
                        print(f"[DEBUG] Gen {i+1}: gBest未初期化、最良pBest (idx={best_pbest_idx}, fitness={pBests_fitness_host[best_pbest_idx]:.3f})を使用")
            else:
                # Spatial topology: GPU上で完結させる（メモリ転送を削減）
                # 【修正】CuPyが使用可能な場合はGPU上で計算、そうでない場合はCPU側で計算
                if use_cupy:
                    # CuPy配列に変換してGPU上で計算（メモリ効率を考慮）
                    # 注意: Numba CUDA配列とCuPy配列は直接変換できないため、CPU経由で変換
                    swarms_host = swarms_gpu.copy_to_host().astype(np.float32)
                    pBests_host = pBests_gpu.copy_to_host().astype(np.float32)
                    pBests_fitness_host = pBests_fitness_gpu.copy_to_host().astype(np.float32)
                    
                    # GPUメモリに転送（CuPy配列として）
                    swarms_gpu_cupy_for_lbest = cp.asarray(swarms_host)
                    pBests_gpu_cupy_for_lbest = cp.asarray(pBests_host)
                    pBests_fitness_gpu_cupy_for_lbest = cp.asarray(pBests_fitness_host)
                    
                    # GPU上で空間的近傍を計算（完全にGPU上で完結）
                    lBests_gpu_cupy = get_spatial_lbest_gpu(swarms_gpu_cupy_for_lbest, pBests_gpu_cupy_for_lbest, pBests_fitness_gpu_cupy_for_lbest, k=5)
                    
                    # Numba CUDA配列に変換（CPU経由）
                    lBests_host = lBests_gpu_cupy.get()
                    lBests_gpu = cuda.to_device(lBests_host.astype(np.float32))
                    
                    # メモリクリーンアップ
                    del swarms_gpu_cupy_for_lbest, pBests_gpu_cupy_for_lbest, pBests_fitness_gpu_cupy_for_lbest, lBests_gpu_cupy
                else:
                    # CuPyが使用できない場合はCPU側で計算（フォールバック）
                    # 重要: 評価に使った粒子位置（swarms_cpu）を使用
                    swarms_host = swarms_gpu.copy_to_host()
                    pBests_host = pBests_gpu.copy_to_host()
                    pBests_fitness_host = pBests_fitness_gpu.copy_to_host()
                    # CPU側で空間的近傍を計算（元コードと完全一致）
                    lBests_host = get_spatial_lbest(swarms_host, pBests_host, pBests_fitness_host, k=5)
                    lBests_gpu = cuda.to_device(lBests_host.astype(np.float32))
            t_lbest_end = time.time()
            
            # 乱数生成（CPU側で生成してGPUに転送）
            # 元コードと完全一致: r1, r2 = np.random.rand(2, num_par, 1) から分割
            t_rand_start = time.time()
            r_combined = np.random.rand(2, num_par, 1)  # 元コードと同じ方法で生成
            r1_cpu = r_combined[0].astype(np.float32)  # 最初の次元をr1として使用
            r2_cpu = r_combined[1].astype(np.float32)  # 2番目の次元をr2として使用
            r1_gpu = cuda.to_device(r1_cpu)
            r2_gpu = cuda.to_device(r2_cpu)
            t_rand_end = time.time()
            
            # GPU上で速度・位置更新
            t_vel_start = time.time()
            update_velocity_position_kernel[blocks_per_grid, threads_per_block](
                swarms_gpu, velocities_gpu, pBests_gpu, lBests_gpu,
                np.float32(w), np.float32(c1), np.float32(c2),
                r1_gpu, r2_gpu
            )
            cuda.synchronize()
            t_vel_end = time.time()
            
            # CPU側にも反映（互換性のため）
            swarms_cpu = swarms_gpu.copy_to_host().astype(np.float64)
            t_update_end = time.time()
            
            if i < 3 or i % 10 == 0:  # 最初の3世代と10世代ごとに出力
                print(f"[PROFILE] Gen {i+1} - 更新: "
                      f"lBest={t_lbest_end-t_lbest_start:.3f}s "
                      f"乱数={t_rand_end-t_rand_start:.3f}s "
                      f"速度/位置={t_vel_end-t_vel_start:.3f}s "
                      f"合計={t_update_end-t_update_start:.3f}s")
        elif use_cupy:
            try:
                # CuPyが使用可能な場合、GPU上で実行
                # 重要: 評価に使った粒子位置（swarms_evaluated_gpu）を使用
                # もしswarms_evaluated_gpuが定義されていない場合は、swarms_gpu_cupyを使用
                if 'swarms_evaluated_gpu' in locals() and swarms_evaluated_gpu is not None:
                    swarms_for_update = swarms_evaluated_gpu
                else:
                    swarms_for_update = swarms_gpu_cupy
                
                t_lbest_start = time.time()
                if topology == 'spatial':
                    # GPU上で空間的近傍を計算（完全にGPU上で完結）
                    lBest_matrix_gpu = get_spatial_lbest_gpu(swarms_for_update, pBests_gpu_cupy, pBests_fitness_gpu_cupy, k=5)
                else:  # global
                    # 【修正】gBestが未初期化の場合は、最初の世代の最良粒子（pBestの中で最良）を使用
                    if 'gBest_gpu_cupy' in locals() and gBest_gpu_cupy is not None and gBest_fitness > -float('inf'):
                        lBest_matrix_gpu = get_global_lbest_gpu(swarms_for_update, gBest_gpu_cupy)
                    else:
                        # 最初の世代では、全粒子のpBestの中で最良のものを使用
                        best_pbest_idx_gpu = int(cp.argmax(pBests_fitness_gpu_cupy))
                        best_pbest_gpu = pBests_gpu_cupy[best_pbest_idx_gpu]
                        lBest_matrix_gpu = cp.tile(best_pbest_gpu, (swarms_for_update.shape[0], 1))
                        if i == 0:
                            print(f"[DEBUG] Gen {i+1}: gBest未初期化、最良pBest (idx={best_pbest_idx_gpu}, fitness={float(pBests_fitness_gpu_cupy[best_pbest_idx_gpu]):.3f})を使用")
                t_lbest_end = time.time()
                
                # 乱数生成もGPU上で実行（高速化、CuPyの乱数生成器を使用）
                # 元コードと一致: 同じ乱数生成方法を使用
                t_rand_start = time.time()
                # CuPyの乱数生成器を使用（GPU上で完結）
                r_combined_gpu = cp.random.rand(2, num_par, 1, dtype=cp.float32)
                r1_gpu = r_combined_gpu[0]  # 最初の次元をr1として使用
                r2_gpu = r_combined_gpu[1]  # 2番目の次元をr2として使用
                t_rand_end = time.time()
                
                # GPU上で速度更新を計算（評価に使った粒子位置を使用、完全にGPU上で完結）
                t_vel_start = time.time()
                # ベクトル化された速度更新（GPU上で並列実行）
                velocities_gpu_cupy = (w * velocities_gpu_cupy + 
                                     c1 * r1_gpu * (pBests_gpu_cupy - swarms_for_update) + 
                                     c2 * r2_gpu * (lBest_matrix_gpu - swarms_for_update))
                
                # GPU上で位置更新（評価に使った粒子位置を更新、インプレース演算）
                swarms_for_update += velocities_gpu_cupy
                t_vel_end = time.time()
                
                # 更新された粒子位置をswarms_gpu_cupyとswarms_cpuに反映
                swarms_gpu_cupy = swarms_for_update
                # CPU側への転送は必要最小限（互換性のため）
                swarms_cpu = swarms_gpu_cupy.get().astype(np.float64)
                t_update_end = time.time()
                
                if i < 3 or i % 10 == 0:  # 最初の3世代と10世代ごとに出力
                    print(f"[PROFILE] Gen {i+1} - CuPy更新: "
                          f"lBest={t_lbest_end-t_lbest_start:.3f}s "
                          f"乱数={t_rand_end-t_rand_start:.3f}s "
                          f"速度/位置={t_vel_end-t_vel_start:.3f}s "
                          f"合計={t_update_end-t_update_start:.3f}s")
            except Exception as e:
                print(f"Warning: CuPy PSO update failed: {e}. Falling back to CPU.")
                import traceback
                traceback.print_exc()
                use_cupy = False
        
        if not use_numba_cuda_updates and not use_cupy:
            # CPU側で実行（Numba CUDAとCuPyの両方が使用できない場合のみ）
            t_lbest_cpu_start = time.time()
            if topology == 'global':
                # 【修正】gBestが未初期化の場合は、最初の世代の最良粒子（pBestの中で最良）を使用
                if gBest_cpu is not None and gBest_fitness > -float('inf'):
                    lBest_matrix_cpu = np.tile(gBest_cpu, (num_par, 1))
                else:
                    # 最初の世代では、全粒子のpBestの中で最良のものを使用
                    best_pbest_idx = int(np.argmax(pBests_fitness_cpu))
                    best_pbest = pBests_cpu[best_pbest_idx]
                    lBest_matrix_cpu = np.tile(best_pbest, (num_par, 1))
                    if i == 0:
                        print(f"[DEBUG] Gen {i+1}: gBest未初期化、最良pBest (idx={best_pbest_idx}, fitness={pBests_fitness_cpu[best_pbest_idx]:.3f})を使用")
            else:
                # Spatial topology（multiprocessing版と同じ実装）
                lBest_matrix_cpu = get_spatial_lbest(swarms_cpu, pBests_cpu, pBests_fitness_cpu, k=5)
            t_lbest_cpu_end = time.time()
            
            t_rand_cpu_start = time.time()
            # 元コードと完全一致: r1, r2 = np.random.rand(2, num_par, 1) から分割
            r_combined = np.random.rand(2, num_par, 1)
            r1_cpu = r_combined[0]  # 最初の次元をr1として使用
            r2_cpu = r_combined[1]  # 2番目の次元をr2として使用
            t_rand_cpu_end = time.time()
            
            t_vel_cpu_start = time.time()
            velocities_cpu = (w * velocities_cpu + 
                            c1 * r1_cpu * (pBests_cpu - swarms_cpu) + 
                            c2 * r2_cpu * (lBest_matrix_cpu - swarms_cpu))
            swarms_cpu += velocities_cpu
            t_vel_cpu_end = time.time()
            
            if i < 3 or i % 10 == 0:  # 最初の3世代と10世代ごとに出力
                print(f"[PROFILE] Gen {i+1} - CPU更新: "
                      f"lBest={t_lbest_cpu_end-t_lbest_cpu_start:.3f}s "
                      f"乱数={t_rand_cpu_end-t_rand_cpu_start:.3f}s "
                      f"速度/位置={t_vel_cpu_end-t_vel_cpu_start:.3f}s")
        
        t_gen_end = time.time()
        if i < 3 or i % 10 == 0:  # 最初の3世代と10世代ごとに出力
            print(f"[PROFILE] Gen {i+1} - 世代全体: {t_gen_end-t_gen_start:.3f}s")

    total_time = time.time() - pso_start_time
    print(f"[PROFILE] PSO全体実行時間: {total_time:.3f}s")
    return gBest_feasible_bn, total_time

# ==========================================
# 9. Main Comparison Loop
# ==========================================
if __name__ == '__main__':
    # Tuned Parameters
    pso_params = {
        'num_par': 100, 'num_gen': 100,
        'convergence_gen': 100,
        'time_limit_sec': 9900, 
        'w_config': (0.9, 0.18), 'c1_config': (3.32, 0.44), 'c2_config': (0.88,3.66),
        'Pd_config': (6.63, 56.4), 
        'Pl_config': (73.9, 56447.8),  
        'Pr_config': (73.9, 56447.8)
    }
    
    constraints = {'delay_multiplier': 3.0, 'loss_constraint': 0.1, 'reliability_constraint': 0.95}
    
    # 実験設定
    node_counts = [1000] 
    num_trials = 1
    
    result_path = savef.create_dir(dir_name="comparison_benchmark_cuda")
    csv_file = f"{result_path}/compare_cuda_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    if CUDA_AVAILABLE:
        try:
            device = cp.cuda.Device()
            compute_cap = device.compute_capability
            meminfo = cp.cuda.runtime.memGetInfo()
            cupy_version = cp.__version__
            cuda_runtime_version = cp.cuda.runtime.runtimeGetVersion()
            major = cuda_runtime_version // 1000
            minor = (cuda_runtime_version % 1000) // 10
            print(f"Using GPU: CuPy {cupy_version}, CUDA Runtime {major}.{minor}, "
                  f"Compute Capability {compute_cap}, "
                  f"Memory: {meminfo[0] / 1024**3:.2f} GB free / {meminfo[1] / 1024**3:.2f} GB total")
        except Exception as e:
            print(f"Warning: GPU information not available: {e}")
    else:
        print("Warning: Running on CPU (CuPy not available)")
        print("Note: For CUDA 11.x, install CuPy with: pip install cupy-cuda11x")
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'NodeCount', 'Trial', 
            'Exact_BN', 'Exact_Time', 
            'Global_BN', 'Global_Time', 
            'Spatial_BN', 'Spatial_Time', 
            'Restart_BN', 'Restart_Time'
        ])
        
        print("=== 4手法 比較ベンチマーク（Numba CUDA + CuPy並列化版）開始 ===")
        
        for N in node_counts:
            print(f"\n--- Node Count: {N} ---")
            
            for t in range(num_trials):
                trial_start = time.time()
                print(f"\n=== Trial {t+1}/{num_trials} ===")
                
                # 1. グラフ生成
                t_graph_gen_start = time.time()
                Graph = create_graph.random_graph(num_node=N)
                if not nx.is_connected(Graph):
                    largest_cc = max(nx.connected_components(Graph), key=len)
                    nodes = list(largest_cc)
                else:
                    nodes = list(Graph.nodes())
                if len(nodes) < 2: continue
                src, dst = random.sample(nodes, 2)
                t_graph_gen_end = time.time()
                print(f"[PROFILE] グラフ生成: {t_graph_gen_end - t_graph_gen_start:.3f}s")
                
                # --- Method 1: Exact (Baseline) - 厳密解法はそのまま ---
                print(f"Trial {t+1} - Exact...", end="\r")
                try:
                    d_path = nx.dijkstra_path(Graph, source=src, target=dst, weight='delay')
                    _, min_d, _, _ = calculate_path_attributes_4d(Graph, d_path)
                    abs_max_delay = min_d * constraints['delay_multiplier']
                    
                    st = time.time()
                    exact_bn = find_optimal_path_by_label_correcting_4d(
                        Graph, src, dst, abs_max_delay, 
                        constraints['loss_constraint'], constraints['reliability_constraint']
                    )
                    exact_time = time.time() - st
                except:
                    exact_bn = -1; exact_time = 0
                
                # --- Method 2: Global PSO (CUDA) ---
                print(f"Trial {t+1} - Global PSO (CUDA)...", end="\r")
                t_glob_start = time.time()
                glob_bn, glob_time = run_pso_cuda(
                    Graph, src, dst, constraints, pso_params, 
                    topology='global', enable_restart=False
                )
                t_glob_end = time.time()
                print(f"[PROFILE] Global PSO実時間: {t_glob_end - t_glob_start:.3f}s (報告時間: {glob_time:.3f}s)")
                
                # --- Method 3: Spatial PSO (CUDA) ---
                print(f"Trial {t+1} - Spatial PSO (CUDA)...", end="\r")
                t_spat_start = time.time()
                spat_bn, spat_time = run_pso_cuda(
                    Graph, src, dst, constraints, pso_params, 
                    topology='spatial', enable_restart=False
                )
                t_spat_end = time.time()
                print(f"[PROFILE] Spatial PSO実時間: {t_spat_end - t_spat_start:.3f}s (報告時間: {spat_time:.3f}s)")

                # --- Method 4: Spatial + Restart (CUDA) ---
                print(f"Trial {t+1} - Restart PSO (CUDA)...", end="\r")
                t_rest_start = time.time()
                rest_bn, rest_time = run_pso_cuda(
                    Graph, src, dst, constraints, pso_params, 
                    topology='spatial', enable_restart=True
                )
                t_rest_end = time.time()
                print(f"[PROFILE] Restart PSO実時間: {t_rest_end - t_rest_start:.3f}s (報告時間: {rest_time:.3f}s)")
                
                trial_end = time.time()
                print(f"[PROFILE] Trial {t+1} 全体: {trial_end - trial_start:.3f}s")
                
                # Log
                print(f"Trial {t+1}: Exact={exact_bn:.2f}({exact_time:.2f}s) | "
                      f"Glob={glob_bn:.2f}({glob_time:.2f}s) | "
                      f"Spat={spat_bn:.2f}({spat_time:.2f}s) | "
                      f"Rest={rest_bn:.2f}({rest_time:.2f}s)")
                
                # CSV Save
                writer.writerow([
                    N, t+1, 
                    exact_bn, exact_time, 
                    glob_bn, glob_time, 
                    spat_bn, spat_time,
                    rest_bn, rest_time
                ])
                # バッファフラッシュ（途中経過を確実に保存）
                f.flush()

    print(f"\n全実験終了。結果は {csv_file} に保存されました。")
