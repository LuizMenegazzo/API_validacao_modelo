@echo off
setlocal

if not exist ".venv\Scripts\python.exe" (
    echo Criando ambiente virtual...
    python -m venv .venv
)

echo Ativando ambiente virtual...
call .venv\Scripts\activate

echo Instalando dependencias...
python -m pip install --upgrade pip
pip install -r requirements.txt

set "BUILD_ROOT=%TEMP%\comparador_modelos_build"
set "DIST_ROOT=%TEMP%\comparador_modelos_dist"
set "FINAL_EXE=%DIST_ROOT%\comparador_modelos.exe"
set "PROJECT_DIST=%CD%\dist_exe"
set "PROJECT_EXE=%PROJECT_DIST%\comparador_modelos.exe"
set "FALLBACK_DIST=%LOCALAPPDATA%\ModelValidationBuilds"
set "FALLBACK_EXE=%FALLBACK_DIST%\comparador_modelos.exe"

echo Limpando diretorios temporarios de build...
if exist "%BUILD_ROOT%" rmdir /s /q "%BUILD_ROOT%"
if exist "%DIST_ROOT%" rmdir /s /q "%DIST_ROOT%"
if not exist "%PROJECT_DIST%" mkdir "%PROJECT_DIST%"
if exist "%PROJECT_EXE%" del /f /q "%PROJECT_EXE%"
if not exist "%FALLBACK_DIST%" mkdir "%FALLBACK_DIST%"
if exist "%FALLBACK_EXE%" del /f /q "%FALLBACK_EXE%"

echo Gerando executavel...
python -m PyInstaller --noconfirm --clean --distpath "%DIST_ROOT%" --workpath "%BUILD_ROOT%" comparador_modelos.spec

if exist "%FINAL_EXE%" (
    copy /y "%FINAL_EXE%" "%PROJECT_EXE%" >nul
    if exist "%PROJECT_EXE%" (
        echo.
        echo Executavel gerado com sucesso:
        echo %PROJECT_EXE%
    ) else (
        copy /y "%FINAL_EXE%" "%FALLBACK_EXE%" >nul
        echo.
        echo Nao foi possivel copiar para %PROJECT_DIST%.
        echo Executavel salvo na pasta alternativa:
        echo %FALLBACK_EXE%
    )
) else (
    echo.
    echo Falha: o executavel nao foi encontrado em %FINAL_EXE%
    exit /b 1
)
