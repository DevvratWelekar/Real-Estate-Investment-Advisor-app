$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$url = "http://127.0.0.1:8501"
$chromeCandidates = @(
    (Join-Path $env:ProgramFiles "Google\Chrome\Application\chrome.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Google\Chrome\Application\chrome.exe"),
    (Join-Path $env:LOCALAPPDATA "Google\Chrome\Application\chrome.exe")
)
$chrome = $chromeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $chrome) {
    $chromeCommand = Get-Command chrome.exe -ErrorAction SilentlyContinue
    if ($chromeCommand) {
        $chrome = $chromeCommand.Source
    }
}

if (-not $chrome) {
    throw "Google Chrome was not found. Install Chrome, then run this script again."
}

$server = $null
try {
    Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 | Out-Null
} catch {
    $server = Start-Process -FilePath "python" `
        -ArgumentList @(
            "-m", "streamlit", "run", "app.py",
            "--server.address", "127.0.0.1",
            "--server.headless", "true",
            "--server.port", "8501"
        ) `
        -WorkingDirectory $projectRoot -NoNewWindow -PassThru

    $deadline = (Get-Date).AddSeconds(60)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        if ($server.HasExited) {
            throw "Streamlit exited before it became ready (exit code $($server.ExitCode))."
        }

        try {
            Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 | Out-Null
            $ready = $true
            break
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }

    if (-not $ready) {
        throw "Streamlit did not become ready at $url within 60 seconds."
    }
}

Start-Process -FilePath $chrome -ArgumentList $url

if ($server) {
    try {
        $server.WaitForExit()
        exit $server.ExitCode
    } finally {
        if (-not $server.HasExited) {
            Stop-Process -Id $server.Id
        }
    }
}
