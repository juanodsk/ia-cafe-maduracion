param([string]$PythonExecutable = 'python')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        & $PythonExecutable -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'No se pudo crear el entorno. Utiliza Python 3.12.' }
    }
    & .\.venv\Scripts\python.exe -m pip install --cache-dir .cache\pip torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'Error al instalar PyTorch CUDA.' }
    & .\.venv\Scripts\python.exe -m pip install --cache-dir .cache\pip -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Error al instalar las dependencias del proyecto.' }
    & .\.venv\Scripts\python.exe -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no disponible")'
} finally { Pop-Location }
