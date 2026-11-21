@echo off
setlocal
rem Allow this downloaded checkout's unsigned scripts in this process only.
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-control-harness.ps1" %*
set "launch_status=%ERRORLEVEL%"
if not "%launch_status%"=="0" (
  echo.
  echo Startup did not complete. See the message above.
  if "%~1"=="" pause
)
exit /b %launch_status%
