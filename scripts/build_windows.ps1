# Playlist Generator Windows Build Script
# Creates a standalone Windows executable using PyInstaller

param(
    [switch]$Install,
    [switch]$Clean,
    [switch]$OneFile
)

$ErrorActionPreference = "Stop"

# Project root
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $ProjectRoot

try {
    # Clean previous builds
    if ($Clean) {
        Write-Host "Cleaning previous builds..."
        if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
        if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
        if (Test-Path "*.spec" -PathType Leaf) { Remove-Item -Force "*.spec" }
    }

    # Install dependencies
    if ($Install) {
        Write-Host "Installing dependencies..."
        python -m pip install --upgrade pip
        python -m pip install -r requirements.txt
        python -m pip install -r requirements-gui.txt
        python -m pip install pyinstaller
    }

    # Verify dependencies
    Write-Host "Checking dependencies..."
    python -c "import PySide6; print(f'PySide6 {PySide6.__version__}')"
    python -c "import yaml; print('PyYAML OK')"

    # Build with PyInstaller
    Write-Host "Building application..."

    $PyInstallerArgs = @(
        "--name", "PlaylistGenerator",
        "--windowed",
        "--noconfirm",
        "--add-data", "config.example.yaml;.",
        "--add-data", "data;data",
        "--hidden-import", "PySide6.QtCore",
        "--hidden-import", "PySide6.QtWidgets",
        "--hidden-import", "PySide6.QtGui",
        "--hidden-import", "yaml",
        "--hidden-import", "platformdirs",
        "--hidden-import", "src.playlist_gui",
        "--hidden-import", "src.playlist_gui.worker",
        "--hidden-import", "src.config_loader",
        "--hidden-import", "src.playlist_generator",
        "--hidden-import", "src.local_library_client",
        "--hidden-import", "src.playlist.pipeline",
        "--hidden-import", "src.features.artifacts",
        "--collect-all", "PySide6",
        "src/playlist_gui/app.py"
    )

    if ($OneFile) {
        $PyInstallerArgs = @("--onefile") + $PyInstallerArgs
    }

    & pyinstaller @PyInstallerArgs

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Build successful!" -ForegroundColor Green
        Write-Host "Output: $ProjectRoot\dist\PlaylistGenerator"
        Write-Host ""
        Write-Host "To run: dist\PlaylistGenerator\PlaylistGenerator.exe"
    } else {
        Write-Host "Build failed!" -ForegroundColor Red
        exit 1
    }
}
finally {
    Pop-Location
}
