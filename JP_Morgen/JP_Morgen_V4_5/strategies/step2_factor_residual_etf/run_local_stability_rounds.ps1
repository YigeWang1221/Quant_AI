param(
    [ValidateSet("all", "round1_base", "round2_regularized")]
    [string]$Round = "all",
    [string]$Seeds = "101,202,303,404,505",
    [switch]$SkipOfflineSweep,
    [switch]$ContinueOnError,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$python = "C:\Users\59386\.conda\envs\Quant311\python.exe"
$script = "D:\quant\quant1\JP_Morgen\JP_Morgen_V4_5\strategies\step2_factor_residual_etf\run_local_stability_rounds.py"

$command = @(
    $script,
    "--round", $Round,
    "--seeds", $Seeds,
    "--python_exe", $python
)

if ($SkipOfflineSweep) {
    $command += "--skip_offline_sweep"
}
if ($ContinueOnError) {
    $command += "--continue_on_error"
}
if ($ExtraArgs.Count -gt 0) {
    $command += "--"
    $command += $ExtraArgs
}

& $python @command
