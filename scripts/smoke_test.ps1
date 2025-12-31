# Smoke test driver for Playlist Generator (Windows)
# Steps:
# 1) Ensure venv exists; create if missing
# 2) Install editable package
# 3) Run pytest
# 4) Run worker protocol smoke (ping + doctor)
# 5) Optional: PyInstaller build when PYINSTALLER=1

$ErrorActionPreference = "Stop"

function Write-Info($msg) {
    Write-Host "[INFO] $msg" -ForegroundColor Cyan
}

function Ensure-Venv {
    if (-Not (Test-Path ".venv")) {
        Write-Info "Creating virtual environment..."
        python -m venv .venv
    }
    Write-Info "Activating virtual environment..."
    .\.venv\Scripts\Activate.ps1
}

function Install-Deps {
    Write-Info "Installing dependencies (editable)..."
    python -m pip install -U pip
    pip install -r requirements.txt
    if (Test-Path "requirements-gui.txt") { pip install -r requirements-gui.txt }
    pip install -e .
}

function Run-Pytest {
    Write-Info "Running pytest..."
    python -m pytest -q
}

function Run-WorkerSmoke {
    Write-Info "Running worker protocol smoke..."
    $script = @"
import json, sys, subprocess, os, time
from pathlib import Path

env = os.environ.copy()
env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
proc = subprocess.Popen([sys.executable, "-m", "playlist_gui.worker"],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True,
                        cwd=str(Path(__file__).resolve().parents[1]),
                        env=env)
try:
    cmds = [
        {"cmd": "ping", "request_id": "smoke-ping", "protocol_version": 1},
        {"cmd": "doctor", "request_id": "smoke-doc", "base_config_path": "config.example.yaml"},
    ]
    for c in cmds:
        proc.stdin.write(json.dumps(c) + "\\n")
    proc.stdin.flush()

    seen = set()
    start = time.time()
    lines = []
    while time.time() - start < 10 and len(seen) < 2:
        line = proc.stdout.readline()
        if not line:
            break
        lines.append(line)
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = evt.get("request_id")
        if evt.get("type") == "done" and rid:
            seen.add(rid)
    if "smoke-ping" not in seen or "smoke-doc" not in seen:
        sys.stderr.write("Worker smoke failed. Output:\\n" + "".join(lines))
        sys.exit(1)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
"@
    $tmp = New-TemporaryFile
    Set-Content -Path $tmp -Value $script -Encoding UTF8
    python $tmp
    Remove-Item $tmp -Force
}

function Run-PyInstaller {
    if ($env:PYINSTALLER -eq "1") {
        Write-Info "PYINSTALLER=1 set; running build script..."
        ./scripts/build_windows.ps1
    } else {
        Write-Info "PYINSTALLER not set; skipping build."
    }
}

Ensure-Venv
Install-Deps
Run-Pytest
Run-WorkerSmoke
Run-PyInstaller
Write-Info "Smoke tests completed successfully."
