# CuPy用のPATH設定スクリプト
# このスクリプトを実行してから、Pythonスクリプトを実行してください

Write-Host "Checking for CUDA installations..."

# CUDA 12.xのパスを探す
$cuda_paths = @(
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.7",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
)

$found_cuda = $null
foreach ($path in $cuda_paths) {
    if (Test-Path $path) {
        $bin_path = Join-Path $path "bin"
        if (Test-Path $bin_path) {
            $nvrtc_dll = Join-Path $bin_path "nvrtc64_120_0.dll"
            if (Test-Path $nvrtc_dll) {
                Write-Host "Found CUDA installation: $path"
                Write-Host "  nvrtc64_120_0.dll found at: $nvrtc_dll"
                $found_cuda = $path
                break
            }
        }
    }
}

if ($found_cuda) {
    $bin_path = Join-Path $found_cuda "bin"
    $current_path = $env:PATH
    
    if ($current_path -notlike "*$bin_path*") {
        Write-Host "Adding $bin_path to PATH for this session..."
        $env:PATH = "$bin_path;$env:PATH"
        Write-Host "PATH updated. You can now run your Python script."
        Write-Host ""
        Write-Host "To make this permanent, add the following to your system PATH:"
        Write-Host "  $bin_path"
    } else {
        Write-Host "PATH already contains $bin_path"
    }
} else {
    Write-Host "CUDA 12.x installation not found."
    Write-Host "Please install CUDA 12.x from: https://developer.nvidia.com/cuda-downloads"
    Write-Host ""
    Write-Host "Alternatively, you can try installing CuPy for CUDA 11.x:"
    Write-Host "  pip install cupy-cuda11x"
}
