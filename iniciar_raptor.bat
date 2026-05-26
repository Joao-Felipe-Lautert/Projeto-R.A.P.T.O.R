@echo off
REM ╔══════════════════════════════════════════════════════════════════════╗
REM ║          R.A.P.T.O.R. Touch - Script de Inicialização               ║
REM ║                    Para Windows                                      ║
REM ╚══════════════════════════════════════════════════════════════════════╝

echo ╔══════════════════════════════════════════╗
echo ║  R.A.P.T.O.R. Touch - Inicializador      ║
echo ╚══════════════════════════════════════════╝

REM Verifica se Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python não está instalado ou não está no PATH.
    echo Instale Python 3.8+ e certifique-se de adicionar ao PATH.
    pause
    exit /b 1
)

echo [OK] Python encontrado

REM Instala dependências
echo.
echo [INSTALANDO] Dependências...
pip install -r requirements.txt

if errorlevel 1 (
    echo [ERRO] Falha ao instalar dependências
    pause
    exit /b 1
)

echo [OK] Dependências instaladas com sucesso

REM Inicia a aplicação
echo.
echo [INICIANDO] R.A.P.T.O.R. Touch...
python raptor_touch.py

echo [INFO] Aplicação encerrada
pause
