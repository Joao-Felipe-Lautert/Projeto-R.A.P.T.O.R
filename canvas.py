"""
JARVIS - Módulo de Canvas de Desenho (Versão Manipulável)
Gerencia o canvas virtual e objetos que podem ser movidos e redimensionados.
"""

import cv2
import numpy as np
from collections import deque


# Paleta de cores estilo JARVIS
COLORS = {
    "draw":    (0, 255, 200),    # Ciano neon
    "erase":   (0, 0, 0),        # Preto (apagar)
    "result":  (0, 220, 255),    # Amarelo-ciano
    "shape":   (100, 255, 100),  # Verde neon (forma corrigida)
    "text":    (200, 255, 255),  # Branco-azulado
    "warning": (0, 100, 255),    # Laranja
    "ui_bg":   (10, 20, 30),     # Fundo escuro
    "ui_line": (0, 180, 180),    # Linha UI
    "select":  (255, 255, 0),    # Amarelo (seleção)
}


class DrawingCanvas:
    """Canvas virtual para desenho e manipulação de objetos."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height

        # Canvas principal (fundo preto transparente)
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Histórico de pontos para suavização
        self.points: list[deque] = [deque(maxlen=512)]
        self.current_stroke: int = 0

        # Todos os traços (para análise)
        self.all_strokes: list[list[tuple]] = []
        self.current_raw_stroke: list[tuple] = []

        self.drawing = False
        self.brush_size = 4
        self.color = COLORS["draw"]

        # Objetos manipuláveis (formas e textos)
        self.objects = []  # Lista de dicionários: {type, params, color, selected}
        self.selected_object_idx = -1

        # Histórico de ações para undo
        self._history: list[np.ndarray] = []

    def save_state(self):
        """Salva estado atual para undo."""
        self._history.append(self.canvas.copy())
        if len(self._history) > 20:
            self._history.pop(0)

    def undo(self):
        """Desfaz última ação."""
        if self._history:
            self.canvas = self._history.pop()
            if self.objects:
                self.objects.pop()

    def start_stroke(self, point: tuple):
        """Inicia um novo traço."""
        self.drawing = True
        self.current_raw_stroke = [point]
        self.points.append(deque(maxlen=512))
        self.current_stroke = len(self.points) - 1
        self.points[self.current_stroke].appendleft(point)

    def add_point(self, point: tuple):
        """Adiciona ponto ao traço atual."""
        if self.drawing and point:
            self.current_raw_stroke.append(point)
            self.points[self.current_stroke].appendleft(point)

            # Desenha linha suavizada
            if len(self.points[self.current_stroke]) >= 2:
                p1 = self.points[self.current_stroke][0]
                p2 = self.points[self.current_stroke][1]
                cv2.line(self.canvas, p1, p2, self.color, self.brush_size,
                         cv2.LINE_AA)

    def end_stroke(self):
        """Finaliza o traço atual."""
        if self.drawing and self.current_raw_stroke:
            self.all_strokes.append(list(self.current_raw_stroke))
        self.drawing = False
        self.current_raw_stroke = []

    def clear(self):
        """Limpa o canvas."""
        self.save_state()
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.points = [deque(maxlen=512)]
        self.current_stroke = 0
        self.all_strokes = []
        self.objects = []
        self.selected_object_idx = -1

    def erase_area(self, center: tuple, radius: int = 30):
        """Apaga uma área circular."""
        cv2.circle(self.canvas, center, radius, (0, 0, 0), -1)
        # Remove objetos próximos se necessário
        self.objects = [obj for obj in self.objects if not self._is_near_object(center, obj, radius)]

    def _is_near_object(self, point, obj, radius):
        """Verifica se um ponto está próximo de um objeto."""
        if obj["type"] == "circle":
            cx, cy = obj["params"]["center"]
            dist = np.sqrt((point[0] - cx)**2 + (point[1] - cy)**2)
            return dist < (obj["params"]["radius"] + radius)
        elif obj["type"] == "rectangle":
            x, y, w, h = obj["params"]["x"], obj["params"]["y"], obj["params"]["w"], obj["params"]["h"]
            return (x - radius < point[0] < x + w + radius) and (y - radius < point[1] < y + h + radius)
        elif obj["type"] == "text":
            x, y = obj["params"]["pos"]
            return np.sqrt((point[0] - x)**2 + (point[1] - y)**2) < radius * 2
        return False

    def add_object(self, obj_type, params, color=COLORS["shape"]):
        """Adiciona um objeto manipulável."""
        self.objects.append({
            "type": obj_type,
            "params": params,
            "color": color,
            "selected": False,
            "scale": 1.0
        })

    def draw_result(self, text, pos, color=COLORS["result"]):
        """Alias para manter compatibilidade com chamadas antigas."""
        self.add_object("text", {"text": text, "pos": pos}, color=color)

    def select_object(self, point):
        """Seleciona o objeto mais próximo de um ponto."""
        self.selected_object_idx = -1
        min_dist = 100
        for i, obj in enumerate(self.objects):
            obj["selected"] = False
            if obj["type"] == "circle":
                cx, cy = obj["params"]["center"]
                dist = np.sqrt((point[0] - cx)**2 + (point[1] - cy)**2)
                if dist < obj["params"]["radius"] + 20:
                    if dist < min_dist:
                        min_dist = dist
                        self.selected_object_idx = i
            elif obj["type"] == "rectangle":
                x, y, w, h = obj["params"]["x"], obj["params"]["y"], obj["params"]["w"], obj["params"]["h"]
                if (x - 20 < point[0] < x + w + 20) and (y - 20 < point[1] < y + h + 20):
                    self.selected_object_idx = i
            elif obj["type"] == "text":
                x, y = obj["params"]["pos"]
                dist = np.sqrt((point[0] - x)**2 + (point[1] - y)**2)
                if dist < 50:
                    self.selected_object_idx = i
        
        if self.selected_object_idx != -1:
            self.objects[self.selected_object_idx]["selected"] = True
            return self.objects[self.selected_object_idx]
        return None

    def move_selected(self, new_pos):
        """Move o objeto selecionado para uma nova posição."""
        if self.selected_object_idx != -1:
            obj = self.objects[self.selected_object_idx]
            if obj["type"] == "circle":
                obj["params"]["center"] = new_pos
            elif obj["type"] == "rectangle":
                obj["params"]["x"] = new_pos[0] - obj["params"]["w"] // 2
                obj["params"]["y"] = new_pos[1] - obj["params"]["h"] // 2
            elif obj["type"] == "text":
                obj["params"]["pos"] = new_pos

    def scale_selected(self, factor):
        """Altera a escala do objeto selecionado."""
        if self.selected_object_idx != -1:
            obj = self.objects[self.selected_object_idx]
            obj["scale"] *= factor
            obj["scale"] = max(0.2, min(obj["scale"], 5.0)) # Limites de escala
            
            if obj["type"] == "circle":
                obj["params"]["radius"] = int(obj["params"]["radius"] * factor)
            elif obj["type"] == "rectangle":
                obj["params"]["w"] = int(obj["params"]["w"] * factor)
                obj["params"]["h"] = int(obj["params"]["h"] * factor)

    def get_canvas_with_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Combina o frame com o canvas e renderiza objetos manipuláveis."""
        # Cria um canvas temporário para renderizar objetos
        temp_canvas = self.canvas.copy()
        
        for obj in self.objects:
            color = COLORS["select"] if obj["selected"] else obj["color"]
            thickness = 3 if obj["selected"] else 2
            
            if obj["type"] == "circle":
                cv2.circle(temp_canvas, obj["params"]["center"], obj["params"]["radius"], color, thickness, cv2.LINE_AA)
            elif obj["type"] == "rectangle":
                x, y, w, h = obj["params"]["x"], obj["params"]["y"], obj["params"]["w"], obj["params"]["h"]
                cv2.rectangle(temp_canvas, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)
            elif obj["type"] == "text":
                cv2.putText(temp_canvas, obj["params"]["text"], obj["params"]["pos"], 
                            cv2.FONT_HERSHEY_DUPLEX, 0.8 * obj["scale"], color, thickness, cv2.LINE_AA)

        # Máscara: pixels não-pretos do canvas
        mask = cv2.cvtColor(temp_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY)

        # Aplica o canvas sobre o frame
        result = frame.copy()
        canvas_area = cv2.bitwise_and(temp_canvas, temp_canvas, mask=mask)
        frame_area = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(mask))
        result = cv2.add(frame_area, canvas_area)

        return result
