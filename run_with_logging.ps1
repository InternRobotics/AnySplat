# Run demo_gradio.py with full logging
$logFile = "app_full_log_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$pythonExe = ".\anysplat_env\Scripts\python.exe"
$script = "demo_gradio.py"

Write-Host "Starting app with logging to: $logFile"
Write-Host "Press Ctrl+C to stop"

# Run Python and redirect both stdout and stderr to log file
& $pythonExe $script *>&1 | Tee-Object -FilePath $logFile
