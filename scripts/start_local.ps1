param(
    [int]$VoicePort = 8000,
    [int]$ConversationPort = 8001
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Missing .venv. Create it with: python -m venv .venv"
}
if (-not (Test-Path (Join-Path $root ".env"))) {
    throw "Missing .env. Copy .env.example to .env and configure the required providers."
}

$env:PYTHONPATH = $root
$importCheck = & $python -c "import fastapi, httpx, pydantic, dotenv, redis"
if ($LASTEXITCODE -ne 0) {
    throw "Python dependencies are incomplete. Install services/conversation/requirements.txt and services/voice/requirements.txt in .venv."
}

function Get-Readiness($port) {
    try {
        return Invoke-RestMethod -Uri "http://127.0.0.1:$port/ready" -TimeoutSec 5
    } catch {
        return $null
    }
}

function Wait-ForReadiness($name, $port, $logPath, $maxAttempts = 60) {
    for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
        $ready = Get-Readiness $port
        if ($ready -and $ready.status -eq "ready") {
            Write-Host "$name is ready on port $port."
            return
        }
        Start-Sleep -Milliseconds 500
    }

    Write-Host "--- $name startup log ---" -ForegroundColor Yellow
    if (Test-Path $logPath) {
        Get-Content $logPath -Tail 40
    }
    throw "$name did not become ready on port $port."
}

function Start-TriageService($name, $module, $port, $stdoutPath, $stderrPath) {
    $ready = Get-Readiness $port
    if ($ready -and $ready.status -eq "ready") {
        Write-Host "$name is already ready on port $port."
        return $false
    }

    $listener = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
    if ($listener) {
        throw "Port $port is already in use, but it is not a ready TriageOS service."
    }

    Start-Process -FilePath $python `
        -ArgumentList "-m", "uvicorn", $module, "--host", "127.0.0.1", "--port", $port `
        -WorkingDirectory $root `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -WindowStyle Hidden | Out-Null
    Write-Host "Starting $name..."
    return $true
}

$logs = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

Start-TriageService "Conversation service" `
    "services.conversation.app.main:app" `
    $ConversationPort `
    (Join-Path $logs "conversation.local.out.log") `
    (Join-Path $logs "conversation.local.err.log") | Out-Null
Wait-ForReadiness "Conversation service" $ConversationPort (Join-Path $logs "conversation.local.err.log")

Start-TriageService "Voice service" `
    "services.voice.app.main:app" `
    $VoicePort `
    (Join-Path $logs "voice.local.out.log") `
    (Join-Path $logs "voice.local.err.log") | Out-Null
Wait-ForReadiness "Voice service" $VoicePort (Join-Path $logs "voice.local.err.log")

Write-Host "TriageOS is ready: http://localhost:$VoicePort" -ForegroundColor Green
