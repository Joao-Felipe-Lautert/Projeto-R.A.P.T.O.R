"""
R.A.P.T.O.R. — Script de Configuração e Instalação
Execute: python setup.py
"""

import subprocess
import sys
import os
import platform


def run(cmd, check=True):
    print(f"  → {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if check and result.returncode != 0:
        print(f"  [AVISO] Comando retornou código {result.returncode}")
    return result.returncode == 0


def check_python():
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("[ERRO] Python 3.10+ é necessário.")
        sys.exit(1)
    print("[OK] Versão do Python compatível.")


def install_packages():
    print("\n[1/3] Instalando pacotes Python...")
    packages = [
        "opencv-python",
        "mediapipe",
        "numpy",
        "sympy",
        "pytesseract",
        "Pillow",
    ]
    pip_cmd = f"{sys.executable} -m pip install " + " ".join(packages)
    run(pip_cmd)
    print("[OK] Pacotes instalados.")


def check_tesseract():
    print("\n[2/3] Verificando Tesseract OCR...")
    result = subprocess.run("tesseract --version", shell=True,
                            capture_output=True, text=True)
    if result.returncode == 0:
        version = result.stdout.split("\n")[0]
        print(f"[OK] Tesseract encontrado: {version}")
    else:
        system = platform.system()
        print("[AVISO] Tesseract não encontrado.")
        print("        O reconhecimento de expressões matemáticas ficará desativado.")
        print()
        if system == "Windows":
            print("  Para instalar no Windows:")
            print("  1. Baixe em: https://github.com/UB-Mannheim/tesseract/wiki")
            print("  2. Instale e adicione ao PATH do sistema")
        elif system == "Linux":
            print("  Para instalar no Linux:")
            print("  sudo apt-get install tesseract-ocr")
        elif system == "Darwin":
            print("  Para instalar no macOS:")
            print("  brew install tesseract")


def check_api_key():
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        print(f"[OK] OPENAI_API_KEY configurada ({key[:8]}...)")
    else:
        print("[INFO] OPENAI_API_KEY não configurada.")
        print("       O assistente funcionará em modo offline (respostas básicas).")
        print()
        print("  Para configurar:")
        if platform.system() == "Windows":
            print("  $env:OPENAI_API_KEY = 'sua-chave-aqui'  (PowerShell)")
            print("  set OPENAI_API_KEY=sua-chave-aqui        (CMD)")
        else:
            print("  export OPENAI_API_KEY='sua-chave-aqui'")


def test_camera():
    print("\n[EXTRA] Testando câmera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("[OK] Câmera (índice 0) disponível.")
            cap.release()
        else:
            print("[AVISO] Câmera índice 0 não disponível.")
            print("        Tente: python raptor.py --camera 1")
    except ImportError:
        print("[AVISO] OpenCV não importado ainda.")


def main():
    print("=" * 50)
    print("  R.A.P.T.O.R. — Configuração do Sistema")
    print("=" * 50)
    print()

    check_python()
    install_packages()
    check_tesseract()
    check_api_key()
    test_camera()

    print()
    print("=" * 50)
    print("  Configuração concluída!")
    print()
    print("  Para iniciar o RAPTOR:")
    print("  python raptor.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
