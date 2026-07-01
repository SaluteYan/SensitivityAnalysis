param(
    [string]$EnvName = "algorithm_py_env",
    [string]$OutputDir = "results/problem21_opmwade_sensitivity_balanced",
    [string]$InitFile = "init_data/PrG21InitData-target_1_05-none.npz",
    [string]$DampingMode = "none",
    [double]$TargetAngle = 1.05,
    [int]$AnchorMaxNfes = 6400,
    [int]$SensitivityMaxNfes = 6400,
    [int]$ParetoMaxNfes = 6400,
    [int]$AnchorRepeats = 8,
    [int]$SensitivityRepeats = 6,
    [int]$ParetoRepeats = 6,
    [int]$ParetoGrid = 4,
    [double]$InitialNpFactor = 18.0,
    [double]$MinNpFactor = 5.0,
    [int]$Workers = 0,
    [int]$ThreadsPerWorker = 1
)

$ErrorActionPreference = "Stop"

if ($Workers -le 0) {
    if ([Environment]::ProcessorCount -ge 8) {
        $Workers = [Environment]::ProcessorCount - 2
    }
    else {
        $Workers = [Math]::Max(1, [Environment]::ProcessorCount - 1)
    }
}

$env:OMP_NUM_THREADS = "$ThreadsPerWorker"
$env:OPENBLAS_NUM_THREADS = "$ThreadsPerWorker"
$env:MKL_NUM_THREADS = "$ThreadsPerWorker"
$env:NUMEXPR_NUM_THREADS = "$ThreadsPerWorker"
$env:VECLIB_MAXIMUM_THREADS = "$ThreadsPerWorker"

if (Get-Command conda -ErrorAction SilentlyContinue) {
    conda activate $EnvName
}

python -m pip install -r requirements.txt
New-Item -ItemType Directory -Force -Path logs | Out-Null
$LogFile = "logs/problem21_paper_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss")
Write-Host "Writing log to $LogFile"
Write-Host "workers=$Workers, threads_per_worker=$ThreadsPerWorker"

python experiments/problem21_opmwade_sensitivity.py `
    --output-dir $OutputDir `
    --init-file $InitFile `
    --damping-mode $DampingMode `
    --target-angle $TargetAngle `
    --anchor-max-nfes $AnchorMaxNfes `
    --sensitivity-max-nfes $SensitivityMaxNfes `
    --pareto-max-nfes $ParetoMaxNfes `
    --anchor-repeats $AnchorRepeats `
    --sensitivity-repeats $SensitivityRepeats `
    --pareto-repeats $ParetoRepeats `
    --workers $Workers `
    --initial-np-factor $InitialNpFactor `
    --min-np-factor $MinNpFactor `
    --pareto-grid $ParetoGrid `
    --enable-late-enhancements `
    2>&1 | Tee-Object -FilePath $LogFile
