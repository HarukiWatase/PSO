"""CUDAとCuPyのバージョンを確認するスクリプト"""
import os
import sys
import subprocess

print("=" * 60)
print("CUDAとCuPyのバージョン確認")
print("=" * 60)

# 1. CUDAのバージョン確認
print("\n[1] CUDAのバージョン確認")
print("-" * 60)

# nvccコマンドで確認
try:
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("nvccコマンドで確認:")
        print(result.stdout)
    else:
        print("nvccコマンドが見つかりません")
except FileNotFoundError:
    print("nvccコマンドが見つかりません（CUDAがインストールされていない可能性）")
except Exception as e:
    print(f"nvccコマンドの実行エラー: {e}")

# CUDA_PATH環境変数を確認
cuda_path = os.environ.get('CUDA_PATH', '')
if cuda_path:
    print(f"\nCUDA_PATH環境変数: {cuda_path}")
    if os.path.exists(cuda_path):
        print(f"  → 存在します")
        # binディレクトリ内のDLLを確認
        bin_path = os.path.join(cuda_path, 'bin')
        if os.path.exists(bin_path):
            dlls = [f for f in os.listdir(bin_path) if f.endswith('.dll')]
            nvrtc_dlls = [f for f in dlls if 'nvrtc' in f.lower()]
            if nvrtc_dlls:
                print(f"  → NVRTC DLLが見つかりました: {nvrtc_dlls}")
            else:
                print(f"  → NVRTC DLLが見つかりません")
    else:
        print(f"  → 存在しません")
else:
    print("CUDA_PATH環境変数が設定されていません")

# 一般的なCUDAインストールパスを確認
print("\n一般的なCUDAインストールパスを確認:")
cuda_install_paths = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
]
for base_path in cuda_install_paths:
    if os.path.exists(base_path):
        versions = [d for d in os.listdir(base_path) if d.startswith('v')]
        if versions:
            print(f"  {base_path}:")
            for version in sorted(versions, reverse=True):
                version_path = os.path.join(base_path, version)
                bin_path = os.path.join(version_path, 'bin')
                if os.path.exists(bin_path):
                    dlls = [f for f in os.listdir(bin_path) if f.endswith('.dll')]
                    nvrtc_dlls = [f for f in dlls if 'nvrtc' in f.lower()]
                    nvrtc_info = f" (NVRTC: {len(nvrtc_dlls)}個)" if nvrtc_dlls else " (NVRTCなし)"
                    print(f"    - {version}{nvrtc_info}")
                    if nvrtc_dlls:
                        print(f"      → {', '.join(nvrtc_dlls[:3])}...")
        else:
            print(f"  {base_path}: バージョンが見つかりません")
    else:
        print(f"  {base_path}: 存在しません")

# 2. CuPyのバージョン確認
print("\n[2] CuPyのバージョン確認")
print("-" * 60)

try:
    import cupy as cp
    print(f"CuPyバージョン: {cp.__version__}")
    
    # CUDAバージョン情報
    try:
        cuda_runtime_version = cp.cuda.runtime.runtimeGetVersion()
        major = cuda_runtime_version // 1000
        minor = (cuda_runtime_version % 1000) // 10
        print(f"CUDA Runtimeバージョン: {major}.{minor}")
    except:
        print("CUDA Runtimeバージョンの取得に失敗")
    
    # GPUデバイス情報
    try:
        device = cp.cuda.Device()
        print(f"GPU Device ID: {device.id}")
        print(f"Compute Capability: {device.compute_capability}")
        
        meminfo = cp.cuda.runtime.memGetInfo()
        print(f"GPU Memory: {meminfo[0] / 1024**3:.2f} GB free / {meminfo[1] / 1024**3:.2f} GB total")
    except Exception as e:
        print(f"GPU情報の取得に失敗: {e}")
    
    # 簡単な動作テスト
    print("\nCuPyの動作テスト:")
    try:
        test_array = cp.array([1.0, 2.0, 3.0])
        result = test_array + 1.0
        cpu_result = cp.asnumpy(result)
        print(f"  → テスト成功: {cpu_result}")
        print("  → CuPyは正常に動作しています")
    except Exception as e:
        print(f"  → テスト失敗: {e}")
        
except ImportError:
    print("CuPyがインストールされていません")
    print("インストール方法:")
    print("  - CUDA 11.x用: pip install cupy-cuda11x")
    print("  - CUDA 12.x用: pip install cupy-cuda12x")
except Exception as e:
    print(f"CuPyのインポートエラー: {e}")
    import traceback
    traceback.print_exc()

# 3. PATH環境変数の確認
print("\n[3] PATH環境変数の確認（CUDA関連）")
print("-" * 60)
path_env = os.environ.get('PATH', '')
cuda_paths_in_path = [p for p in path_env.split(os.pathsep) if 'cuda' in p.lower()]
if cuda_paths_in_path:
    print("PATHに含まれるCUDA関連パス:")
    for p in cuda_paths_in_path[:10]:  # 最初の10個のみ表示
        print(f"  - {p}")
    if len(cuda_paths_in_path) > 10:
        print(f"  ... 他 {len(cuda_paths_in_path) - 10}個")
else:
    print("PATHにCUDA関連のパスが見つかりません")

print("\n" + "=" * 60)
print("確認完了")
print("=" * 60)
