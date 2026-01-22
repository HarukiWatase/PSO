# CuPyのNVRTCエラー解決方法

## 問題
CuPyが`nvrtc64_120_0.dll`を見つけられないエラーが発生しています。

## 解決方法

### 方法1: PATH環境変数を設定（推奨）

1. **一時的な解決（現在のセッションのみ）**
   ```powershell
   # PowerShellで実行
   .\fix_cupy_path.ps1
   python watase_34_cuda.py
   ```

2. **永続的な解決（システム全体）**
   - Windowsの設定 → システム → 詳細情報 → システムの詳細設定
   - 「環境変数」をクリック
   - 「システム環境変数」の「Path」を選択して「編集」
   - CUDA 12.xのbinディレクトリを追加（例：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`）

### 方法2: コード内でPATHを設定

`watase_34_cuda.py`の先頭に以下を追加：

```python
import os
import sys

# CUDA 12.xのbinディレクトリをPATHに追加
cuda_bin_paths = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
    # ... 他のバージョン
]

for path in cuda_bin_paths:
    if os.path.exists(path) and path not in os.environ.get('PATH', ''):
        os.environ['PATH'] = path + os.pathsep + os.environ.get('PATH', '')
```

### 方法3: CUDA 11.x用のCuPyを使用

CUDA 12.xがインストールされていない場合、CUDA 11.x用のCuPyを使用できます：

```powershell
.\venv_cuda\Scripts\Activate.ps1
pip uninstall cupy-cuda12x
pip install cupy-cuda11x
```

### 方法4: CuPyなしで実行（現在の状態）

現在のコードは、CuPyが使えない場合でもCPU側で動作するように実装されています。
ただし、PSO更新処理がCPU側で実行されるため、パフォーマンスが低下します。

## 確認方法

CuPyが正しく動作しているか確認：

```powershell
.\venv_cuda\Scripts\python.exe check_cupy.py
```

## 注意事項

- CuPy 13.6.0はCUDA 12.x用です
- CUDA 11.xを使用している場合は、`cupy-cuda11x`をインストールしてください
- NVRTCはCuPyの一部機能（カスタムカーネルなど）に必要ですが、基本的な配列操作には不要です
