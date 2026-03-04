# 🦖 R.A.P.T.O.R. — Real-time AI Processing & Tracking Operational Response

O **R.A.P.T.O.R.** é um sistema avançado de **visão computacional** e **reconhecimento de gestos** que cria uma interface interativa e intuitiva. Utilizando a câmera do seu computador, o sistema permite desenhar no ar, reconhecer formas geométricas e resolver expressões matemáticas em tempo real.

---

## 🚀 Funcionalidades Principais

*   **✍️ Desenho Virtual:** Utilize o dedo indicador para desenhar em um canvas digital sobreposto ao vídeo da câmera.
*   **📐 Reconhecimento Geométrico:** Detecta e corrige automaticamente formas como círculos, quadrados, triângulos, pentágonos e muito mais.
*   **📊 Cálculos em Tempo Real:** Calcula automaticamente área, perímetro, raio e outras propriedades das formas detectadas.
*   **🔢 Solucionador Matemático:** Escreva expressões matemáticas no ar (ex: `15 * 3 =`) e o RAPTOR exibirá o resultado instantaneamente.
*   **🎙️ Ativação por Voz:** Inicie o sistema através do comando de voz "Olá RAPTOR".
*   **🧹 Controles por Gestos:** Limpe o canvas ou inicie análises apenas abrindo ou fechando a mão.

---

## 🛠️ Requisitos do Sistema

Este projeto foi desenvolvido e otimizado para as seguintes especificações:

*   **Linguagem:** Python **3.11** (Recomendado)
*   **Sistema Operacional:** Windows, Linux ou macOS.
*   **Hardware:** Webcam funcional e microfone (para comandos de voz).
*   **Dependência Externa:** [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (necessário para reconhecimento matemático).

---

## 📦 Instalação e Configuração

### 1. Clonar o Repositório
```bash
git clone https://github.com/seu-usuario/Project-R.A.P.T.O.R.git
cd Project-R.A.P.T.O.R
```

### 2. Instalar Dependências Python
Recomendamos o uso de um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 3. Instalar Tesseract OCR
O Tesseract é essencial para a leitura de números e símbolos desenhados.

*   **Windows:** Baixe o instalador em [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) e adicione o diretório de instalação ao seu PATH.
*   **Linux (Ubuntu/Debian):** `sudo apt-get install tesseract-ocr`
*   **macOS:** `brew install tesseract`

---

## 🎮 Como Utilizar

### Inicialização
Você pode iniciar o sistema diretamente ou via ativação por voz:
*   **Direto:** `python raptor.py`
*   **Voz:** `python voice_activation.py` (Diga "Olá RAPTOR" para começar)

### Controles por Gestos
| Gesto | Ação |
| :--- | :--- |
| 👌 **Gesto de Pinça** | Modo Desenho (mova para riscar) |
| ✌️ **Indicador + Médio** | Borracha (apaga traços específicos) |
| 🖐️ **Mão Aberta (0.6s)** | Iniciar Análise (Formas/Matemática) |

### Atalhos de Teclado
*   `Z`: Desfazer última ação.
*   `C`: Limpar canvas.
*   `A`: Forçar análise imediata.
*   `Q` ou `Esc`: Sair do programa.

---

## 📂 Estrutura do Projeto

| Arquivo | Descrição |
| :--- | :--- |
| `raptor.py` | Núcleo principal do sistema e interface de vídeo. |
| `hand_tracker.py` | Motor de rastreamento de mãos via MediaPipe. |
| `canvas.py` | Gerenciamento do canvas de desenho e estados. |
| `shape_recognizer.py` | Lógica de detecção de formas geométricas. |
| `math_recognizer.py` | Processamento de OCR e resolução matemática. |
| `voice_activation.py` | Script de escuta para ativação por voz. |

---

## 🤝 Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir *Issues* ou enviar *Pull Requests*. Para mudanças maiores, por favor, abra uma discussão primeiro.

---

## 🤝 Colaboradores

Este projeto contou com a colaboração de:
- [Felipe Prestes Belusso](https://github.com/FelipeBelusso)
  
Demais contribuições são bem-vindas! Sinta-se à vontade para abrir Issues ou enviar Pull Requests. Para mudanças maiores, por favor, abra uma discussão primeiro.

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

*"Sempre às ordens, senhor."* — **R.A.P.T.O.R.**
