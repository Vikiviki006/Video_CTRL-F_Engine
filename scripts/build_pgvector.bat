@echo off
REM Build + install the pgvector PostgreSQL extension for the local PostgreSQL 18 install.
REM Required because the local server has no `vector` extension (0 rows in
REM pg_available_extensions), so `video_frames.embedding vector(1152)` cannot be created.

setlocal

set WORK=C:\Users\vikik\AppData\Local\Temp\opencode\pgvector-build
set PGROOT=C:\Program Files\PostgreSQL\18
set REPO=https://github.com/pgvector/pgvector.git

echo === [1/5] locate MSVC toolchain ===
set VSWHERE=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe
if not exist "%VSWHERE%" (
  echo FATAL: vswhere.exe not found
  exit /b 1
)
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VSPATH=%%i
if "%VSPATH%"=="" (
  echo FATAL: no VC.Tools.x86.x64 installation found
  exit /b 1
)
echo VSPATH=%VSPATH%

echo === [2/5] clone pgvector ===
if exist "%WORK%" rmdir /s /q "%WORK%"
mkdir "%WORK%" 2>nul
git clone --depth 1 --branch v0.8.1 "%REPO%" "%WORK%"
if errorlevel 1 (
  echo clone of v0.8.1 failed, trying default branch
  git clone --depth 1 "%REPO%" "%WORK%"
  if errorlevel 1 exit /b 1
)

echo === [3/5] enter MSVC x64 environment ===
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
  echo FATAL: vcvars64 failed
  exit /b 1
)
where nmake
if errorlevel 1 (
  echo FATAL: nmake not on PATH after vcvars64
  exit /b 1
)

echo === [4/5] build ===
pushd "%WORK%"
set PG_CONFIG=%PGROOT%\bin\pg_config.exe
"%PG_CONFIG%" --version
nmake /F Makefile.win
if errorlevel 1 (
  echo FATAL: build failed
  popd
  exit /b 1
)

echo === [5/5] install ===
nmake /F Makefile.win install
if errorlevel 1 (
  echo FATAL: install failed
  popd
  exit /b 1
)
popd

echo === PGVECTOR BUILD COMPLETE ===
