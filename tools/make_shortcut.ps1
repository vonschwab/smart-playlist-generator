param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$PythonW = (Get-Command pythonw.exe -ErrorAction SilentlyContinue).Source,
    [string]$ShortcutName = "Playlist Generator.lnk"
)

if (-not $PythonW) {
    $PythonW = (Get-Command python.exe -ErrorAction Stop).Source
}

$desktop = [Environment]::GetFolderPath('Desktop')
$shortcutPath = Join-Path $desktop $ShortcutName

$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut($shortcutPath)
$sc.TargetPath = $PythonW
$sc.Arguments = "-m playlist_gui.app"
$sc.WorkingDirectory = $ProjectRoot
$sc.IconLocation = "$PythonW,0"
$sc.Description = "Launch Playlist Generator GUI"
$sc.Save()

Write-Output ("Created: " + $shortcutPath)
