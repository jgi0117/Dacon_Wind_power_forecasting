param(
    [Parameter(Mandatory = $true)]
    [string[]]$Models,
    [ValidateSet('cpu', 'cuda')]
    [string]$Device = 'cpu',
    [switch]$ReuseCompleted,
    [switch]$EvaluationOnly,
    [switch]$DryRun,
    [int]$Seed = 42,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PipelineArgs
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repoRoot '.venv313\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Missing .venv313; run scripts\setup_env.ps1 first.'
}

$arguments = @(
    (Join-Path $PSScriptRoot 'run_models.py'),
    '--models'
) + $Models + @(
    '--device', $Device,
    '--seed', $Seed
)
if ($ReuseCompleted) { $arguments += '--reuse-completed' }
if ($EvaluationOnly) { $arguments += '--evaluation-only' }
if ($DryRun) { $arguments += '--dry-run' }
if ($PipelineArgs) { $arguments += @('--pipeline-args') + $PipelineArgs }

& $python @arguments
exit $LASTEXITCODE
