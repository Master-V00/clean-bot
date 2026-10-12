# One-way sync: Local -> Server
# Usage: .\sync_to_server.ps1

$SERVER = "YOUR_SERVER_IP"
$PORT = ""
$REMOTE_PATH = "/workspace/Clean bot"
$LOCAL_PATH = ""

# Files/folders to EXCLUDE from sync
$EXCLUDE_LIST = @(
    ".git",
    "venv",
    "node_modules",
    "__pycache__",
    "temp",
    "temp_chapters",
    "output",
    "weights",
    ".\app\services\models",
    "app/vendor",
    "app/__pycache__",
    "app/data",
    "credentials/",
    "*.pid",
    "*.pyc"
)

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host " Sync: Local -> Server" -ForegroundColor Cyan
Write-Host " $SERVER`:$PORT$REMOTE_PATH" -ForegroundColor DarkGray
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Build tar-based archive with exclusions, then scp
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$archiveName = "sync_$timestamp.tar.gz"
$localTemp = "$env:TEMP\$archiveName"

# Create exclude file
$excludeFile = "$env:TEMP\sync_excludes.txt"
$EXCLUDE_LIST | Set-Content -Path $excludeFile

Write-Host "Creating archive (excluding: $($EXCLUDE_LIST -join ', '))..." -ForegroundColor Yellow

# Use tar to create archive (built into Windows 10+)
$tarExcludes = ""
foreach ($ex in $EXCLUDE_LIST) {
    $tarExcludes += " --exclude=`"$ex`""
}

$tarCmd = "tar -czf `"$localTemp`" -C `"$LOCAL_PATH`" $tarExcludes ."
cmd /c $tarCmd 2>$null

if (-not (Test-Path $localTemp)) {
    Write-Host "Failed to create archive" -ForegroundColor Red
    exit 1
}

$archiveSize = (Get-Item $localTemp).Length / 1MB
Write-Host "Archive created: $([math]::Round($archiveSize, 2)) MB" -ForegroundColor Green

# Upload archive
Write-Host "Uploading to server..." -ForegroundColor Yellow
scp -P $PORT -o StrictHostKeyChecking=no $localTemp "root@${SERVER}:/tmp/$archiveName"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Upload failed" -ForegroundColor Red
    Remove-Item $localTemp -ErrorAction SilentlyContinue
    exit 1
}

Write-Host "Extracting on server..." -ForegroundColor Yellow
$extractCmd = "ssh -p $PORT -o StrictHostKeyChecking=no root@$SERVER " +
    "`"cd '$REMOTE_PATH' && tar -xzf /tmp/$archiveName && rm /tmp/$archiveName`""

cmd /c $extractCmd

# Cleanup
Remove-Item $localTemp -ErrorAction SilentlyContinue
Remove-Item $excludeFile -ErrorAction SilentlyContinue

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Sync complete!" -ForegroundColor Green
} else {
    Write-Host "Extraction failed on server" -ForegroundColor Red
}
