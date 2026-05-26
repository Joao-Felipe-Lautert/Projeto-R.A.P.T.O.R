# R.A.P.T.O.R. Touch - Sistema de Desenho Geométrico

## 📋 Descrição

**R.A.P.T.O.R. Touch** é uma aplicação desktop Python que permite desenho geométrico interativo com reconhecimento automático de formas, cálculo de área e perímetro. Desenvolvida a partir do projeto original RAPTOR, mas adaptada para interface touch (mouse/trackpad) sem necessidade de câmera ou rastreamento de mãos.

### Funcionalidades Principais

- ✏️ **Desenho com Mouse/Touch** - Interface intuitiva para desenhar formas
- 🔍 **Reconhecimento Automático de Formas** - Detecta círculos, retângulos, triângulos, linhas e polígonos
- 📐 **Cálculo de Área e Perímetro** - Calcula automaticamente medidas em centímetros
- 🎨 **Paleta de Cores RAPTOR** - Interface com identidade visual futurista (Azul Ciano + Magenta)
- 🖌️ **Ferramentas de Edição** - Borracha, desfazer, limpar canvas
- 📊 **Exibição de Dimensões** - Mostra medidas ao lado de cada forma detectada

## 🚀 Instalação

### Pré-requisitos

- Python 3.8+
- pip (gerenciador de pacotes Python)

### Passos de Instalação

1. **Clone ou extraia o projeto:**
```bash
cd RAPTOR-Touch-Python
```

2. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

## 🎮 Como Usar

### Iniciar a Aplicação

```bash
python raptor_touch.py
```

A janela da aplicação será aberta com o canvas preto e os botões de controle na parte inferior.

### Controles

#### Interface Gráfica (Botões)
- **Desenhar** - Ativa modo de desenho (padrão)
- **Apagar** - Ativa modo de borracha
- **Analisar** - Detecta e corrige formas desenhadas
- **Desfazer** - Desfaz última ação
- **Limpar** - Limpa todo o canvas

#### Atalhos de Teclado
- **D** - Modo de desenho
- **E** - Modo de apagar
- **A** - Analisar formas
- **Z** - Desfazer
- **C** - Limpar canvas
- **Q ou ESC** - Sair da aplicação

#### Mouse/Touch
- **Clique e arraste** - Desenhar ou apagar (depende do modo)
- **Clique em botão** - Ativar função

## 📐 Formas Suportadas

A aplicação reconhece automaticamente as seguintes formas:

| Forma | Medidas Calculadas |
|-------|-------------------|
| **Círculo** | Raio, Diâmetro, Área, Circunferência |
| **Retângulo/Quadrado** | Largura, Altura, Área, Perímetro |
| **Triângulo** | Lados, Área (Heron), Perímetro, Tipo (Equilátero/Isósceles/Escaleno) |
| **Elipse** | Semi-eixos, Área, Perímetro |
| **Linha** | Comprimento, Ângulo |
| **Polígonos** | Área, Perímetro |

## 🎨 Paleta de Cores

A interface segue a identidade visual da logo RAPTOR:

- **Azul Ciano** (#00FFFF) - Cor primária de desenho
- **Magenta** (#FF00FF) - Cor de destaque/seleção
- **Azul Profundo** (#0033FF) - Fundo
- **Preto** (#000000) - Fundo do canvas
- **Verde** (#00FF00) - Formas corrigidas

## 📁 Estrutura do Projeto

```
RAPTOR-Touch-Python/
├── raptor_touch.py          # Aplicação principal
├── raptor_config.py         # Configurações e constantes
├── shape_recognizer.py      # Módulo de reconhecimento de formas
├── shape_corrector.py       # Módulo de correção e desenho de formas
├── requirements.txt         # Dependências Python
├── README.md               # Este arquivo
└── RAPTOR_logo.jpg         # Logo da aplicação (referência)
```

## 🔧 Configuração

Edite `raptor_config.py` para personalizar:

- **Dimensões do canvas** - `CANVAS_WIDTH`, `CANVAS_HEIGHT`
- **Tamanho do pincel** - `BRUSH_SIZE`
- **Raio da borracha** - `ERASER_RADIUS`
- **Cores** - Dicionário `COLORS`
- **Conversão de pixels** - `PIXELS_PER_CM`

## 📊 Exemplo de Uso

1. Inicie a aplicação: `python raptor_touch.py`
2. Desenhe um círculo no canvas
3. Clique em **Analisar**
4. A aplicação detectará o círculo, corrigirá a forma e exibirá:
   - Raio em cm
   - Diâmetro em cm
   - Área em cm²
   - Circunferência em cm

## 🐛 Troubleshooting

### Erro: "ModuleNotFoundError: No module named 'cv2'"
**Solução:** Instale opencv-python
```bash
pip install opencv-python
```

### Erro: "ModuleNotFoundError: No module named 'numpy'"
**Solução:** Instale numpy
```bash
pip install numpy
```

### A janela não aparece
**Solução:** Verifique se você tem um gerenciador de janelas funcionando. Em sistemas headless, use:
```bash
export DISPLAY=:0
python raptor_touch.py
```

## 📝 Notas Técnicas

- **Conversão de Medidas**: O sistema usa 37.8 pixels por centímetro (padrão 96 DPI)
- **Reconhecimento**: Usa OpenCV com análise de contornos e aproximação poligonal
- **Histórico**: Mantém até 20 ações para desfazer
- **Suavização**: Aplica dilatação para conectar traços próximos

## 🔄 Diferenças da Versão Original

| Recurso | Original | Touch |
|---------|----------|-------|
| Entrada | Câmera + Rastreamento de mãos | Mouse/Touch |
| Gestos | Pinça, 2 dedos, mão aberta | Cliques e botões |
| Interface | Overlay em tempo real | Botões dedicados |
| Plataforma | Windows/Linux | Windows/Mac/Linux |
| Dependências | MediaPipe, Tesseract | OpenCV, NumPy |

## 📄 Licença

Este projeto é baseado no RAPTOR original e mantém a mesma filosofia educacional.

## 👨‍💻 Desenvolvimento

Desenvolvido com Python 3.8+ usando:
- **OpenCV** - Processamento de imagem e reconhecimento de formas
- **NumPy** - Operações numéricas e manipulação de arrays

## 💡 Próximas Melhorias

- [ ] Salvar/carregar desenhos em arquivo
- [ ] Exportar imagens com medidas
- [ ] Modo de medição manual com dimensões desenhadas
- [ ] Suporte a múltiplas cores de desenho
- [ ] Histórico visual de operações

---

**Versão:** 1.0  
**Data:** 2026  
**Autor:** R.A.P.T.O.R. Team
