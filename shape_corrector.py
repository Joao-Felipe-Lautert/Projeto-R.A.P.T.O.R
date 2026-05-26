"""
╔══════════════════════════════════════════════════════════════════════╗
║          R.A.P.T.O.R. — Módulo de Correção de Formas                ║
║                    Versão Touch (Sem Câmera)                         ║
╚══════════════════════════════════════════════════════════════════════╝

Redesenha formas tortas de maneira geométrica perfeita sobre o canvas.
"""

import cv2
import numpy as np
from raptor_config import COLORS


class ShapeCorrector:
    """Corrige e renderiza formas geométricas no canvas."""

    def __init__(self, canvas_width: int = 1200, canvas_height: int = 700):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

    def correct_and_draw(self, canvas: np.ndarray, shape, erase_original: bool = True) -> np.ndarray:
        """
        Corrige e desenha uma forma no canvas.
        """
        result = canvas.copy()

        if erase_original and hasattr(shape, 'contour') and shape.contour is not None:
            self._erase_original(result, shape.contour)

        self._draw_corrected(result, shape)
        self._draw_measurements(result, shape)
        return result

    def _erase_original(self, canvas: np.ndarray, contour: np.ndarray):
        """Apaga o rascunho original."""
        mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.dilate(mask, kernel)
        canvas[mask > 0] = [0, 0, 0]

    def _draw_corrected(self, canvas: np.ndarray, shape):
        """Desenha a forma corrigida."""
        t = 3
        color = COLORS["shape"]
        st = shape.shape_type

        if st == "circle":
            cx, cy = shape.params["center"]
            r = shape.params["radius"]
            cv2.circle(canvas, (cx, cy), r, color, t, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 4, color, -1, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy), (cx + r, cy), color, 1, cv2.LINE_AA)

        elif st == "rectangle":
            x, y, w, h = shape.params["x"], shape.params["y"], shape.params["w"], shape.params["h"]
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, t, cv2.LINE_AA)
            self._draw_corner_marks(canvas, x, y, w, h, color)

        elif st == "triangle":
            pts = np.array(shape.params["points"], dtype=np.int32)
            cv2.polylines(canvas, [pts], True, color, t, cv2.LINE_AA)

        elif st == "ellipse":
            center = tuple(shape.params["center"])
            axes = tuple(shape.params["axes"])
            angle = shape.params["angle"]
            cv2.ellipse(canvas, center, axes, angle, 0, 360, color, t, cv2.LINE_AA)

        elif st == "polygon":
            pts = np.array(shape.params["points"], dtype=np.int32)
            cv2.polylines(canvas, [pts], True, color, t, cv2.LINE_AA)

    def _draw_corner_marks(self, canvas, x, y, w, h, color, size=12):
        """Desenha marcas nos cantos do retângulo."""
        corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        for (cx, cy), (dx, dy) in zip(corners, dirs):
            cv2.line(canvas, (cx, cy), (cx + dx * size, cy), color, 2, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy), (cx, cy + dy * size), color, 2, cv2.LINE_AA)

    def _draw_measurements(self, canvas: np.ndarray, shape):
        """Desenha as medidas da forma."""
        if not hasattr(shape, 'description') or not shape.description:
            return

        lines = shape.description.split("\n")
        pos = self._get_label_position(shape)
        x, y = pos

        # Fundo do painel
        max_w = max(len(l) for l in lines) * 11 + 10
        box_h = len(lines) * 22 + 10
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x - 5, y - 18), (x + max_w, y + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

        # Desenha texto
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (x, y + i * 22),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, COLORS["text"], 1, cv2.LINE_AA)

    def _get_label_position(self, shape):
        """Determina a posição do rótulo."""
        if shape.shape_type == "circle":
            cx, cy = shape.params["center"]
            return (cx + shape.params["radius"] + 15, cy)
        elif shape.shape_type == "rectangle":
            return (shape.params["x"] + shape.params["w"] + 15, shape.params["y"])
        elif shape.shape_type == "ellipse":
            cx, cy = shape.params["center"]
            return (cx + 50, cy)
        return (50, 50)
