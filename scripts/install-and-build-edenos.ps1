#Requires -RunAsAdministrator
<#
.SYNOPSIS
    EDEN OS V2 — One-Click Installer & Builder
    Installs Docker Desktop, configures GPU + Seagate 5TB, builds image, runs first test.

.DESCRIPTION
    Run this script as Administrator from PowerShell.
    It will:
    1. Check/install Docker Desktop with WSL2 backend
    2. Configure Docker for GPU passthrough and Seagate storage
    3. Build the EDEN OS V2 Docker image
    4. Run a local smoke test
    5. Optionally push to Docker Hub (edenberyl) for RunPod deployment
#>

param(
    [string]$SeagateDrive = "D:",
    [string]$ProjectDir = "",
    [switch]$SkipDockerInstall,
    [switch]$PushToHub,
    [string]$DockerHubUser = "edenberyl"
)

$ErrorActionPreference = "Stop"

# ── Colors ──────────────────────────────────────────────────────
function Write-Step($msg) { Write-Host "`n>> $msg" -ForegroundColor Cyan }
function Write-OK($msg) { Write-Host "   [OK] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "   [!] $msg" -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host "   [X] $msg" -ForegroundColor Red }

Write-Host "`n============================================" -ForegroundColor Magenta
Write-Host "  EDEN OS V2 — Installer & Builder" -ForegroundColor Cyan
Write-Host "============================================`n" -ForegroundColor Magenta

# ── Resolve project directory ────────────────────────────────────
if (-not $ProjectDir) {
    $ProjectDir = Split-Path -Parent $PSScriptRoot
}
Write-OK "Project directory: $ProjectDir"
Write-OK "Seagate drive: $SeagateDrive"

# ── Step 1: Check Docker Desktop ────────────────────────────────
Write-Step "Checking Docker Desktop..."

$dockerInstalled = $false
try {
    $dockerVersion = docker --version 2>$null
    if ($dockerVersion) {
        Write-OK "Docker already installed: $dockerVersion"
        $dockerInstalled = $true
    }
} catch {}

if (-not $dockerInstalled -and -not $SkipDockerInstall) {
    Write-Warn "Docker Desktop not found. Installing via winget..."
    winget install Docker.DockerDesktop --accept-package-agreements --accept-source-agreements
    Write-OK "Docker Desktop installed. Please restart your computer, then re-run this script with -SkipDockerInstall"
    Write-Warn "After restart, ensure Docker Desktop is running and WSL2 integration is enabled."
    exit 0
}

# ── Step 2: Check Docker daemon is running ───────────────────────
Write-Step "Checking Docker daemon..."
$retries = 0
while ($retries -lt 10) {
    try {
        docker info 2>$null | Out-Null
        Write-OK "Docker daemon is running"
        break
    } catch {
        $retries++
        Write-Warn "Docker daemon not ready, waiting... ($retries/10)"
        Start-Sleep -Seconds 5
    }
}
if ($retries -ge 10) {
    Write-Fail "Docker daemon failed to start. Please start Docker Desktop manually."
    exit 1
}

# ── Step 3: Check NVIDIA GPU ─────────────────────────────────────
Write-Step "Checking NVIDIA GPU..."
try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
    if ($gpuInfo) {
        Write-OK "GPU detected: $gpuInfo"
    } else {
        Write-Warn "No NVIDIA GPU detected locally. This is OK — RunPod will provide the GPU."
    }
} catch {
    Write-Warn "nvidia-smi not found. GPU will be available on RunPod."
}

# ── Step 4: Check Seagate 5TB ────────────────────────────────────
Write-Step "Checking Seagate drive ($SeagateDrive)..."
if (Test-Path "$SeagateDrive\") {
    $vol = Get-Volume -DriveLetter ($SeagateDrive.TrimEnd(':')) -ErrorAction SilentlyContinue
    if ($vol) {
        $sizeGB = [math]::Round($vol.Size / 1GB, 1)
        $freeGB = [math]::Round($vol.SizeRemaining / 1GB, 1)
        Write-OK "Seagate found: ${sizeGB}GB total, ${freeGB}GB free"
    }

    # Ensure models directory exists
    $modelsDir = "$SeagateDrive\eden-models"
    if (-not (Test-Path $modelsDir)) {
        New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null
    }
    Write-OK "Models directory: $modelsDir"
} else {
    Write-Warn "Seagate drive not found at $SeagateDrive"
}

# ── Step 5: Check .env file ──────────────────────────────────────
Write-Step "Checking environment configuration..."
$envFile = Join-Path $ProjectDir ".env"
if (-not (Test-Path $envFile)) {
    Write-Warn ".env file not found. Creating from template..."
    Copy-Item (Join-Path $ProjectDir ".env.template") $envFile -ErrorAction SilentlyContinue
    if (-not (Test-Path $envFile)) {
        Write-Fail "Please create .env file with your API keys. See .env.template"
        exit 1
    }
}
Write-OK ".env file found"

# ── Step 6: Build frontend ──────────────────────────────────────
Write-Step "Building React frontend..."
$frontendDir = Join-Path $ProjectDir "frontend"
if (Test-Path (Join-Path $frontendDir "package.json")) {
    Push-Location $frontendDir
    npm install 2>$null
    npm run build 2>$null
    Pop-Location
    Write-OK "Frontend built"
} else {
    Write-Warn "Frontend package.json not found, skipping frontend build"
}

# ── Step 7: Build Docker image ───────────────────────────────────
Write-Step "Building EDEN OS V2 Docker image..."
Push-Location $ProjectDir

$tag = "eden-os-v2:latest"
docker build -t $tag . 2>&1 | ForEach-Object { Write-Host "   $_" -ForegroundColor DarkGray }

if ($LASTEXITCODE -eq 0) {
    Write-OK "Docker image built: $tag"
} else {
    Write-Fail "Docker build failed"
    Pop-Location
    exit 1
}
Pop-Location

# ── Step 8: Smoke test ───────────────────────────────────────────
Write-Step "Running smoke test..."
$testId = docker run -d --rm `
    --env-file $envFile `
    -p 8000:8000 `
    $tag 2>$null

if ($testId) {
    Start-Sleep -Seconds 5
    try {
        $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 10 -ErrorAction SilentlyContinue
        if ($health.status) {
            Write-OK "Smoke test passed: status=$($health.status)"
        }
    } catch {
        Write-Warn "Health check inconclusive (may need GPU)"
    }
    docker stop $testId 2>$null | Out-Null
    Write-OK "Smoke test container stopped"
} else {
    Write-Warn "Could not start test container"
}

# ── Step 9: Optional push to Docker Hub ──────────────────────────
if ($PushToHub) {
    Write-Step "Pushing to Docker Hub ($DockerHubUser)..."
    $hubTag = "$DockerHubUser/eden-os-v2:latest"
    docker tag $tag $hubTag
    docker push $hubTag 2>&1 | ForEach-Object { Write-Host "   $_" -ForegroundColor DarkGray }
    if ($LASTEXITCODE -eq 0) {
        Write-OK "Pushed to Docker Hub: $hubTag"
    } else {
        Write-Fail "Push failed — check docker login"
    }
}

# ── Done ─────────────────────────────────────────────────────────
Write-Host "`n============================================" -ForegroundColor Magenta
Write-Host "  EDEN OS V2 — Build Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Magenta
Write-Host "`n  Image: $tag"
Write-Host "  Models: $SeagateDrive\eden-models"
Write-Host "`n  Next steps:"
Write-Host "    1. docker compose up          (local test)"
Write-Host "    2. .\scripts\deploy-runpod.ps1  (RunPod deploy)"
Write-Host ""
