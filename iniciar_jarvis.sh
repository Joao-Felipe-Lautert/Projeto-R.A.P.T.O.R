#!/bin/bash

echo ""
echo "============================================"
echo "  J.A.R.V.I.S. - Iniciando..."
echo "============================================"
echo ""

# Verifica Python
if ! command -v python3 &> /dev/null; then
    echo "[ERRO] Python3 não encontrado."
    exit 1
fi

# Instala dependências
echo "[INFO] Verificando dependências..."
pip3 install -r requirements.txt --quiet 2>/dev/null || \
    python3 -m pip install -r requirements.txt --quiet

echo "[INFO] Iniciando JARVIS..."
echo ""
python3 jarvis.py "$@"
