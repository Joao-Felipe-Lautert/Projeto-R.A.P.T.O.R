"""
JARVIS - Módulo de Canvas de Desenho
Gerencia o canvas virtual onde o usuário desenha com o dedo.
"""

import cv2
import numpy as np
from collections import deque


# Paleta de cores estilo JARVIS (tons de ciano/azul neon)
COLORS = {
    "draw":    (0, 255, 200),    # Ciano neon
    "erase":   (0, 0, 0),        # Preto (apagar)
    "result":  (0, 220, 255),    # Amarelo-ciano
    "shape":   (100, 255, 100),  # Verde neon (forma corrigida)
    "text":    (200, 255, 255),  # Branco-azulado
    "warning": (0, 100, 255),    # Laranja
    "ui_bg":   (10, 20, 30),     # Fundo escuro
    "ui_line": (0, 180, 180),    # Linha UI
}


class DrawingCanvas:
    """Canvas virtual para desenho com o dedo."""

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

        # Resultado da análise
        self.result_text: str = ""
        self.result_pos: tuple = (0, 0)
        self.corrected_shapes: list = []  # [(tipo, params, cor)]

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
            self.result_text = ""
            self.corrected_shapes.clear()

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
        self.result_text = ""
        self.corrected_shapes.clear()

    def erase_area(self, center: tuple, radius: int = 30):
        """Apaga uma área circular."""
        cv2.circle(self.canvas, center, radius, (0, 0, 0), -1)

    def draw_result(self, text: str, position: tuple, color=None):
        """Desenha o resultado da análise no canvas."""
        if color is None:
            color = COLORS["result"]
        self.result_text = text
        self.result_pos = position

    def draw_corrected_shape(self, shape_type: str, params: dict):
        """Desenha a forma geométrica corrigida sobre o canvas."""
        overlay = self.canvas.copy()
        color = COLORS["shape"]
        thickness = 3

        if shape_type == "circle":
            cx, cy = params["center"]
            r = params["radius"]
            cv2.circle(overlay, (cx, cy), r, color, thickness, cv2.LINE_AA)

        elif shape_type == "rectangle":
            x, y, w, h = params["x"], params["y"], params["w"], params["h"]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness,
                          cv2.LINE_AA)

        elif shape_type == "triangle":
            pts = np.array(params["points"], dtype=np.int32)
            cv2.polylines(overlay, [pts], True, color, thickness, cv2.LINE_AA)

        elif shape_type == "line":
            p1, p2 = params["p1"], params["p2"]
            cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)

        elif shape_type == "polygon":
            pts = np.array(params["points"], dtype=np.int32)
            cv2.polylines(overlay, [pts], True, color, thickness, cv2.LINE_AA)

        # Blend com transparência
        cv2.addWeighted(overlay, 0.7, self.canvas, 0.3, 0, self.canvas)

    def get_all_points_flat(self) -> list[tuple]:
        """Retorna todos os pontos desenhados como lista plana."""
        flat = []
        for stroke in self.all_strokes:
            flat.extend(stroke)
        return flat

    def get_canvas_with_overlay(self, frame: np.ndarray,
                                alpha: float = 0.6) -> np.ndarray:
        """
        Combina o frame da câmera com o canvas de desenho.
        Retorna o frame composto.
        """
        # Máscara: pixels não-pretos do canvas
        mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY)

        # Aplica o canvas sobre o frame
        result = frame.copy()
        canvas_area = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        frame_area = cv2.bitwise_and(result, result,
                                     mask=cv2.bitwise_not(mask))
        result = cv2.add(frame_area, canvas_area)

        # Renderiza texto de resultado
        if self.result_text:
            self._draw_result_text(result)

        return result

    def _draw_result_text(self, frame: np.ndarray):
        """Renderiza o texto de resultado com estilo JARVIS."""
        lines = self.result_text.split("\n")
        x, y = self.result_pos

        for i, line in enumerate(lines):
            pos_y = y + i * 32
            # Sombra
            cv2.putText(frame, line, (x + 2, pos_y + 2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 3,
                        cv2.LINE_AA)
            # Texto principal
            cv2.putText(frame, line, (x, pos_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, COLORS["result"], 2,
                        cv2.LINE_AA)
