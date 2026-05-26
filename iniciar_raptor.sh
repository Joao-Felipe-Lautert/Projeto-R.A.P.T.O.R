#!/bin/bash

# ╔══════════════════════════════════════════════════════════════════════╗
# ║          R.A.P.T.O.R. Touch - Script de Inicialização               ║
# ║                    Para Linux/Mac                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

echo "╔══════════════════════════════════════════╗"
echo "║  R.A.P.T.O.R. Touch - Inicializador      ║"
echo "╚══════════════════════════════════════════╝"

# Verifica se Python está instalado
if ! command -v python3 &> /dev/null; then
    echo "[ERRO] Python 3 não está instalado."
    echo "Instale Python 3.8+ e tente novamente."
    exit 1
fi

echo "[OK] Python 3 encontrado"

# Verifica se pip está instalado
if ! command -v pip3 &> /dev/null; then
    echo "[ERRO] pip3 não está instalado."
    echo "Instale pip3 e tente novamente."
    exit 1
fi

echo "[OK] pip3 encontrado"

# Instala dependências
echo ""
echo "[INSTALANDO] Dependências..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "[OK] Dependências instaladas com sucesso"
else
    echo "[ERRO] Falha ao instalar dependências"
    exit 1
fi

# Inicia a aplicação
echo ""
echo "[INICIANDO] R.A.P.T.O.R. Touch..."
python3 raptor_touch.py

echo "[INFO] Aplicação encerrada"
