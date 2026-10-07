@echo off
rem Starts the MedProof analysis API and the website, then opens the site in a browser.
rem
rem   start.bat            development servers (hot reload)
rem   start.bat preview    production build: prerendered landing, minified bundles
rem
rem Set MEDPROOF_NO_BROWSER=1 to skip opening the browser.

setlocal
cd /d "%~dp0"

set "MODE=%~1"
if "%MODE%"=="" set "MODE=dev"
if /i not "%MODE%"=="dev" if /i not "%MODE%"=="preview" (
  echo Unknown mode "%MODE%". Use: start.bat [dev^|preview]
  exit /b 1
)

rem ---- Python: prefer a project virtualenv, fall back to python on PATH
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=%CD%\.venv\Scripts\python.exe"
if not defined PY if exist "backend\.venv\Scripts\python.exe" set "PY=%CD%\backend\.venv\Scripts\python.exe"
if not defined PY (
  where python >nul 2>nul || (
    echo Python 3.11+ was not found. Install it from https://www.python.org/downloads/ and run this again.
    exit /b 1
  )
  set "PY=python"
)

pushd backend
"%PY%" -c "import fastapi, uvicorn, sse_starlette, multipart, diskcache, fhir.resources, requests, medproof.api.app" >nul 2>nul
if errorlevel 1 (
  echo Installing the analysis API dependencies...
  "%PY%" -m pip install -e ".[services]" || (popd & echo Python dependency install failed. & exit /b 1)
)
"%PY%" -c "import torchxrayvision" >nul 2>nul
if errorlevel 1 (
  echo Note: torch and torchxrayvision are not installed, so uploaded chest films get no reader findings.
  echo       Sample cases work regardless. For live reads: "%PY%" -m pip install -e "backend[ml]"
)
popd

rem ---- Node and pnpm
where node >nul 2>nul || (
  echo Node.js 20+ was not found. Install it from https://nodejs.org/ and run this again.
  exit /b 1
)
set "PNPM=pnpm"
where pnpm >nul 2>nul || set "PNPM=npx --yes pnpm@9"

if not exist "node_modules\.pnpm" (
  echo Installing website dependencies...
  call %PNPM% install || (echo Website dependency install failed. & exit /b 1)
)

rem ---- Servers, each in its own window so their logs stay readable
rem Keys (GROQ_KEY_*, MEDGEMMA_URL, HF_TOKEN) come from .env at the repo root. Without it the notes
rem and report stages fall back to template-only output.
set "ENVFILE="
if exist ".env" (
  set "ENVFILE=--env-file ..\.env"
) else (
  echo Note: no .env found, so notes analysis and the written report run without language models.
)
start "MedProof API (port 8000)" /d "%CD%\backend" cmd /k ""%PY%" -m uvicorn medproof.api.app:app --port 8000 %ENVFILE%"

if /i "%MODE%"=="preview" (
  echo Building the website...
  pushd web
  call %PNPM% run build || (popd & echo Website build failed. & exit /b 1)
  popd
  set "PORT=4173"
  start "MedProof website (port 4173)" /d "%CD%\web" cmd /k "%PNPM% exec vite preview --port 4173 --strictPort"
) else (
  set "PORT=5173"
  start "MedProof website (port 5173)" /d "%CD%\web" cmd /k "%PNPM% exec vite --port 5173 --strictPort"
)

echo Waiting for the website on http://localhost:%PORT% ...
powershell -NoProfile -Command "$u='http://localhost:%PORT%/'; for($i=0;$i -lt 60;$i++){ try { Invoke-WebRequest $u -UseBasicParsing -TimeoutSec 2 | Out-Null; exit 0 } catch { Start-Sleep -Milliseconds 500 } }; exit 1"
if errorlevel 1 (
  echo The website did not respond within 30 seconds. Check the "MedProof website" window for errors.
  exit /b 1
)

echo.
echo   Website       http://localhost:%PORT%/
echo   Workstation   http://localhost:%PORT%/read
echo   API           http://localhost:8000/docs
echo.
echo Close the two server windows to stop everything.
rem Run the sample cases once so the same uploads during a demo hit the cached checks.
start "MedProof warm-up" /min cmd /c ""%PY%" scripts\warm_demo.py"
echo Warming the four sample cases in the background (about a minute the first time).
if not "%MEDPROOF_NO_BROWSER%"=="1" start "" "http://localhost:%PORT%/"
endlocal
