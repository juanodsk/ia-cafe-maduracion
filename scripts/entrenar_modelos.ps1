param(
    [ValidateSet('download', 'train', 'evaluate')][string]$Phase = 'train',
    [string]$Output = 'experimentos\transfer_01',
    [int]$Trials = 12,
    [int]$Epochs = 12,
    [int]$Folds = 5,
    [int]$BatchSize = 64,
    [int]$Workers = 4
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { throw 'Ejecuta primero scripts\instalar_entorno.ps1.' }
    & .\.venv\Scripts\python.exe -u -m cafe_ml.experiment $Phase --output $Output --trials $Trials --epochs $Epochs --folds $Folds --batch-size $BatchSize --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw 'El experimento se interrumpió. Conserva sus archivos para revisar y reanudar.' }
} finally { Pop-Location }
