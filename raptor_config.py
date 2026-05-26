"""
╔══════════════════════════════════════════════════════════════════════╗
║          R.A.P.T.O.R. — Configuração de Cores e Constantes          ║
║                    Versão Touch (Sem Câmera)                         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import math

# ──────────────────────────────────────────────────────────────────────
#  Paleta de Cores (Baseada na Logo RAPTOR)
# ──────────────────────────────────────────────────────────────────────

COLORS = {
    # Cores primárias (Neon futurista)
    "cyan": (0, 255, 255),          # Azul ciano brilhante
    "magenta": (255, 0, 255),       # Magenta/Rosa neon
    
    # Cores de fundo
    "dark_blue": (0, 51, 255),      # Azul profundo
    "black": (0, 0, 0),             # Preto profundo
    "white": (255, 255, 255),       # Branco brilhante
    
    # Cores para UI
    "dark_bg": (10, 14, 30),        # Fundo muito escuro
    "light_text": (224, 224, 255),  # Texto claro
    
    # Cores para desenho
    "draw": (0, 255, 255),          # Ciano para desenho
    "erase": (0, 0, 0),             # Preto para apagar
    "shape": (0, 255, 0),           # Verde para formas corrigidas
    "text": (255, 255, 255),        # Branco para texto
    "accent": (255, 0, 255),        # Magenta para destaque
    "result": (0, 220, 255),        # Amarelo-ciano para resultados
    "ui_bg": (10, 20, 30),          # Fundo UI
    "ui_line": (0, 180, 180),       # Linha UI
    "select": (255, 255, 0),        # Amarelo para seleção
}

# ──────────────────────────────────────────────────────────────────────
#  Configurações da Aplicação
# ──────────────────────────────────────────────────────────────────────

CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 700

BRUSH_SIZE = 4
ERASER_RADIUS = 35

# Conversão de pixels para cm (96 DPI padrão)
PIXELS_PER_CM = 37.8

# Configurações de reconhecimento
MIN_CONTOUR_AREA = 500
CIRCULARITY_THRESHOLD = 0.82

# Histórico de undo
MAX_HISTORY_STEPS = 20

# ──────────────────────────────────────────────────────────────────────
#  Funções Utilitárias de Conversão
# ──────────────────────────────────────────────────────────────────────

def pixels_to_cm(pixels: float) -> float:
    """Converte pixels para centímetros."""
    return pixels / PIXELS_PER_CM

def pixels_square_to_cm_square(pixels_square: float) -> float:
    """Converte pixels² para cm²."""
    return pixels_square / (PIXELS_PER_CM ** 2)

def calculate_circle_area(radius_cm: float) -> float:
    """Calcula área de um círculo em cm²."""
    return math.pi * radius_cm * radius_cm

def calculate_circle_perimeter(radius_cm: float) -> float:
    """Calcula perímetro de um círculo em cm."""
    return 2 * math.pi * radius_cm

def calculate_rectangle_area(width_cm: float, height_cm: float) -> float:
    """Calcula área de um retângulo em cm²."""
    return width_cm * height_cm

def calculate_rectangle_perimeter(width_cm: float, height_cm: float) -> float:
    """Calcula perímetro de um retângulo em cm."""
    return 2 * (width_cm + height_cm)

def calculate_triangle_area(a_cm: float, b_cm: float, c_cm: float) -> float:
    """Calcula área de um triângulo usando fórmula de Heron em cm²."""
    s = (a_cm + b_cm + c_cm) / 2
    area_squared = s * (s - a_cm) * (s - b_cm) * (s - c_cm)
    return math.sqrt(max(area_squared, 0))

def calculate_triangle_perimeter(a_cm: float, b_cm: float, c_cm: float) -> float:
    """Calcula perímetro de um triângulo em cm."""
    return a_cm + b_cm + c_cm

def calculate_ellipse_area(a_cm: float, b_cm: float) -> float:
    """Calcula área de uma elipse em cm²."""
    return math.pi * a_cm * b_cm

def calculate_ellipse_perimeter(a_cm: float, b_cm: float) -> float:
    """Calcula perímetro aproximado de uma elipse usando fórmula de Ramanujan."""
    if a_cm == b_cm:
        return 2 * math.pi * a_cm
    h = ((a_cm - b_cm) / (a_cm + b_cm)) ** 2
    return math.pi * (a_cm + b_cm) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))
