# Copyright (c) 2026 Yota Yamamoto
# SPDX-License-Identifier: MIT

param(
    [string]$PythonExe = "python",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$buildScript = Join-Path $repoRoot "build_windows.py"

$args = @($buildScript)
if ($SkipTests) {
    $args += "--skip-tests"
}

& $PythonExe @args
if ($LASTEXITCODE -ne 0) {
    throw "Build wrapper failed (exit=$LASTEXITCODE)."
}
