@echo off
title R.A.P.T.O.R.
color 0B
echo.
echo  ============================================
echo   R.A.P.T.O.R. - Iniciando...
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

echo [INFO] Iniciando RAPTOR...
echo.
python raptor.py %*

pause
