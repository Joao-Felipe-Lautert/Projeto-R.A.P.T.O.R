# J.A.R.V.I.S. — Just A Rather Very Intelligent System

> *"Bem-vindo ao futuro, senhor."*

Sistema de visão computacional estilo Tony Stark que permite desenhar no ar com a mão, reconhecer formas geométricas, realizar cálculos matemáticos e conversar com uma IA — tudo em tempo real via webcam.

---

## Funcionalidades

| Funcionalidade | Descrição |
|---|---|
| **Desenho no ar** | Use o dedo indicador como caneta virtual |
| **Reconhecimento de formas** | Detecta círculos, retângulos, triângulos, elipses, polígonos |
| **Cálculos automáticos** | Área, perímetro, raio, lados — exibidos automaticamente |
| **Correção de formas** | Redesenha formas tortas de maneira geometricamente perfeita |
| **Expressões matemáticas** | Reconhece contas desenhadas e coloca o resultado ao lado do `=` |
| **Assistente de IA** | Responde perguntas por digitação usando GPT-4 |
| **Interface HUD** | Visual futurista estilo JARVIS com painel de informações |

---

## Requisitos do Sistema

- Python 3.10 ou superior
- Webcam funcional
- Tesseract OCR (para reconhecimento de expressões matemáticas)
- Chave de API OpenAI (opcional, para respostas de IA completas)

---

## Instalação

### 1. Clone ou extraia o projeto

```bash
cd jarvis
```

### 2. Instale as dependências Python

```bash
pip install -r requirements.txt
```

### 3. Instale o Tesseract OCR

**Windows:**
```
Baixe em: https://github.com/UB-Mannheim/tesseract/wiki
Instale e adicione ao PATH do sistema
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### 4. Configure a chave da OpenAI (opcional)

```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY = "sua-chave-aqui"

# Linux / macOS
export OPENAI_API_KEY="sua-chave-aqui"
```

> **Sem a chave OpenAI:** o assistente funciona em modo local com respostas básicas (hora, data, cálculos simples). Com a chave, responde qualquer pergunta via GPT-4.

---

## Como Executar

```bash
python jarvis.py
```

**Opções:**
```bash
python jarvis.py --camera 0    # câmera padrão (índice 0)
python jarvis.py --camera 1    # segunda câmera
```

---

## Controles e Gestos

### Gestos com a Mão

| Gesto | Ação |
|---|---|
| ☝️ **1 dedo** (indicador) | Modo desenho — mova o dedo para desenhar |
| ✌️ **2 dedos** (indicador + médio) | Borracha — apaga onde o dedo passa |
| 🖐️ **Mão aberta** (todos os dedos) | Analisa o desenho atual (segure 0.6s) |
| ✊ **Punho fechado** | Limpa todo o canvas (segure 0.8s) |

### Teclado

| Tecla | Ação |
|---|---|
| `Z` | Desfazer última ação |
| `C` | Limpar canvas |
| `A` | Forçar análise imediata |
| `Enter` | Enviar pergunta ao assistente |
| `Backspace` | Apagar caractere digitado |
| `Esc` | Limpar campo de texto |
| `Q` ou `Esc` (sem texto) | Sair do programa |

---

## Como Usar

### Desenhando e Calculando Formas

1. **Levante apenas o indicador** para entrar no modo de desenho
2. **Mova o dedo** para desenhar a forma desejada (círculo, quadrado, triângulo, etc.)
3. **Abra a mão completamente** e **segure por ~0.6 segundos**
4. O JARVIS irá:
   - Detectar a forma desenhada
   - **Corrigir** a forma (redesenhar geometricamente perfeita)
   - Exibir **área, perímetro e medidas** ao lado da forma

### Calculando Expressões Matemáticas

1. **Desenhe uma expressão** como `25 + 37 =` no ar com o dedo
2. **Abra a mão** para analisar
3. O resultado aparecerá **ao lado do sinal de igual**

### Conversando com o Assistente

1. **Digite sua pergunta** no campo de texto (painel direito)
2. Pressione **Enter** para enviar
3. O JARVIS responderá no painel de chat

---

## Estrutura do Projeto

```
jarvis/
├── jarvis.py           # Arquivo principal — execute este
├── hand_tracker.py     # Rastreamento de mão (MediaPipe)
├── canvas.py           # Canvas de desenho virtual
├── shape_recognizer.py # Reconhecimento de formas geométricas
├── shape_corrector.py  # Correção e renderização de formas
├── math_recognizer.py  # Reconhecimento de expressões matemáticas
├── ai_assistant.py     # Assistente de IA (OpenAI GPT)
├── requirements.txt    # Dependências Python
└── README.md           # Este arquivo
```

---

## Formas Reconhecidas

| Forma | Medidas Calculadas |
|---|---|
| **Círculo** | Raio, diâmetro, área (πr²), circunferência (2πr) |
| **Elipse** | Semi-eixos, área (πab), perímetro (Ramanujan) |
| **Retângulo** | Largura, altura, área (l×h), perímetro (2l+2h) |
| **Quadrado** | Lado, área (l²), perímetro (4l) |
| **Triângulo** | Lados, tipo (equilátero/isósceles/escaleno/retângulo), área (Heron), perímetro |
| **Pentágono** | Área, perímetro |
| **Hexágono** | Área, perímetro |
| **Linha** | Comprimento, ângulo |

---

## Dicas para Melhor Reconhecimento

- **Iluminação:** Use ambiente bem iluminado, de preferência com luz frontal
- **Fundo:** Prefira fundo escuro ou neutro atrás da mão
- **Velocidade:** Desenhe devagar e com movimentos suaves
- **Feche os traços:** Para formas fechadas (círculo, retângulo), conecte o início ao fim
- **Distância:** Mantenha a mão a ~40-60 cm da câmera

---

## Solução de Problemas

**Câmera não abre:**
```bash
python jarvis.py --camera 1  # tente outro índice
```

**Mão não detectada:**
- Verifique a iluminação
- Certifique-se que a mão está completamente visível na câmera

**Tesseract não encontrado:**
- Instale conforme as instruções acima
- No Windows, adicione ao PATH: `C:\Program Files\Tesseract-OCR`

**OpenAI não responde:**
- Verifique se a variável `OPENAI_API_KEY` está definida
- O assistente funciona offline com respostas básicas

---

## Tecnologias Utilizadas

| Tecnologia | Uso |
|---|---|
| **OpenCV** | Captura de vídeo, processamento de imagem, renderização |
| **MediaPipe** | Detecção e rastreamento de landmarks da mão |
| **NumPy** | Operações matriciais e geométricas |
| **SymPy** | Avaliação de expressões matemáticas simbólicas |
| **Tesseract OCR** | Reconhecimento de texto/números desenhados |
| **OpenAI GPT-4** | Respostas inteligentes do assistente |

---

*"Sempre às ordens, senhor."* — J.A.R.V.I.S.
