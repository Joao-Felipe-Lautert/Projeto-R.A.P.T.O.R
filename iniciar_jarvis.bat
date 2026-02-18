@echo off
title J.A.R.V.I.S.
color 0B
echo.
echo  ============================================
echo   J.A.R.V.I.S. - Iniciando...
echo  ============================================
echo.

REM Verifica Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado. Instale em python.org
    pause
    exit /b 1
)

REM Instala dependencias se necessario
echo [INFO] Verificando dependencias...
python -m pip install -r requirements.txt --quiet

echo [INFO] Iniciando JARVIS...
echo.
python jarvis.py %*

pause
