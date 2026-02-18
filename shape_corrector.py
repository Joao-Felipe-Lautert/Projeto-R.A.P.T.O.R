"""
JARVIS - Módulo de Correção de Formas
Redesenha formas tortas de maneira geométrica perfeita sobre o canvas.
"""

import cv2
import numpy as np
import math
from shape_recognizer import ShapeResult


# Cores estilo JARVIS
COLOR_CORRECTED = (0, 255, 180)    # Verde-ciano neon
COLOR_ORIGINAL  = (40, 40, 40)     # Cinza escuro (apaga original)
COLOR_LABEL     = (0, 220, 255)    # Amarelo-ciano
COLOR_MEASURE   = (180, 255, 180)  # Verde claro


class ShapeCorrector:
    """Redesenha formas corrigidas no canvas."""

    def __init__(self, canvas_width: int = 1280, canvas_height: int = 720):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

    def correct_and_draw(self, canvas: np.ndarray,
                          shape: ShapeResult,
                          erase_original: bool = True) -> np.ndarray:
        """
        Apaga a forma original (se solicitado) e desenha a versão corrigida.
        Retorna o canvas modificado.
        """
        result = canvas.copy()

        if erase_original and shape.contour is not None:
            self._erase_original(result, shape.contour)

        self._draw_corrected(result, shape)
        self._draw_measurements(result, shape)

        return result

    # ------------------------------------------------------------------ #
    #  Apagar original                                                     #
    # ------------------------------------------------------------------ #

    def _erase_original(self, canvas: np.ndarray, contour: np.ndarray):
        """Apaga a região do contorno original."""
        mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        # Dilata um pouco para garantir limpeza completa
        kernel = np.ones((8, 8), np.uint8)
        mask = cv2.dilate(mask, kernel)
        canvas[mask > 0] = [0, 0, 0]

    # ------------------------------------------------------------------ #
    #  Desenhar forma corrigida                                            #
    # ------------------------------------------------------------------ #

    def _draw_corrected(self, canvas: np.ndarray, shape: ShapeResult):
        """Desenha a forma corrigida com estilo futurista."""
        t = 3  # espessura
        color = COLOR_CORRECTED

        if shape.shape_type == "circle":
            cx, cy = shape.params["center"]
            r = shape.params["radius"]
            # Círculo principal
            cv2.circle(canvas, (cx, cy), r, color, t, cv2.LINE_AA)
            # Ponto central
            cv2.circle(canvas, (cx, cy), 4, color, -1, cv2.LINE_AA)
            # Linha de raio
            cv2.line(canvas, (cx, cy), (cx + r, cy), color, 1, cv2.LINE_AA)

        elif shape.shape_type == "ellipse":
            center = (int(shape.params["center"][0]),
                      int(shape.params["center"][1]))
            axes = (int(shape.params["axes"][0]),
                    int(shape.params["axes"][1]))
            angle = shape.params["angle"]
            cv2.ellipse(canvas, center, axes, angle, 0, 360, color, t,
                        cv2.LINE_AA)
            cv2.circle(canvas, center, 4, color, -1, cv2.LINE_AA)

        elif shape.shape_type == "rectangle":
            x = shape.params["x"]
            y = shape.params["y"]
            w = shape.params["w"]
            h = shape.params["h"]
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, t,
                          cv2.LINE_AA)
            # Cantos decorativos
            self._draw_corner_marks(canvas, x, y, w, h, color)

        elif shape.shape_type == "triangle":
            pts = np.array(shape.params["points"], dtype=np.int32)
            cv2.polylines(canvas, [pts], True, color, t, cv2.LINE_AA)
            # Marca os vértices
            for pt in pts:
                cv2.circle(canvas, tuple(pt), 5, color, -1, cv2.LINE_AA)

        elif shape.shape_type == "line":
            p1 = tuple(map(int, shape.params["p1"]))
            p2 = tuple(map(int, shape.params["p2"]))
            cv2.line(canvas, p1, p2, color, t, cv2.LINE_AA)
            # Pontas
            cv2.circle(canvas, p1, 5, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, p2, 5, color, -1, cv2.LINE_AA)

        elif shape.shape_type in ("polygon", "triangle"):
            pts = np.array(shape.params["points"], dtype=np.int32)
            cv2.polylines(canvas, [pts], True, color, t, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(canvas, tuple(pt), 5, color, -1, cv2.LINE_AA)

    def _draw_corner_marks(self, canvas, x, y, w, h, color, size=12):
        """Desenha marcas nos cantos de retângulos (estilo HUD)."""
        corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        for (cx, cy), (dx, dy) in zip(corners, dirs):
            cv2.line(canvas, (cx, cy), (cx + dx * size, cy), color, 2,
                     cv2.LINE_AA)
            cv2.line(canvas, (cx, cy), (cx, cy + dy * size), color, 2,
                     cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    #  Rótulos de medidas                                                  #
    # ------------------------------------------------------------------ #

    def _draw_measurements(self, canvas: np.ndarray, shape: ShapeResult):
        """Desenha as medidas da forma no canvas."""
        lines = shape.description.split("\n")
        if not lines:
            return

        # Posição baseada no bounding box da forma
        pos = self._get_label_position(shape)
        x, y = pos

        # Fundo semi-transparente
        max_w = max(len(l) for l in lines) * 10 + 20
        box_h = len(lines) * 22 + 10
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x - 5, y - 18),
                      (x + max_w, y + box_h), (5, 15, 25), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

        # Borda do painel
        cv2.rectangle(canvas, (x - 5, y - 18),
                      (x + max_w, y + box_h), COLOR_CORRECTED, 1)

        for i, line in enumerate(lines):
            color = COLOR_LABEL if i == 0 else COLOR_MEASURE
            font_scale = 0.65 if i == 0 else 0.55
            thickness = 2 if i == 0 else 1
            cv2.putText(canvas, line, (x, y + i * 22),
                        cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness,
                        cv2.LINE_AA)

    def _get_label_position(self, shape: ShapeResult) -> tuple:
        """Calcula posição do rótulo de medidas."""
        if shape.shape_type == "circle":
            cx, cy = shape.params["center"]
            r = shape.params["radius"]
            x = min(cx + r + 15, self.canvas_width - 200)
            y = max(cy - 40, 20)
            return (x, y)

        elif shape.shape_type == "ellipse":
            cx, cy = shape.params["center"]
            ax = shape.params["axes"][0]
            x = min(cx + ax + 15, self.canvas_width - 200)
            y = max(cy - 40, 20)
            return (x, y)

        elif shape.shape_type == "rectangle":
            x = shape.params["x"] + shape.params["w"] + 15
            y = shape.params["y"]
            x = min(x, self.canvas_width - 200)
            return (x, y)

        elif shape.shape_type in ("triangle", "polygon"):
            pts = np.array(shape.params["points"])
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.min(pts[:, 1])) - 15
            return (max(cx - 60, 10), max(cy - 10, 20))

        elif shape.shape_type == "line":
            p1 = shape.params["p1"]
            p2 = shape.params["p2"]
            mx = (p1[0] + p2[0]) // 2 + 10
            my = (p1[1] + p2[1]) // 2 - 10
            return (mx, my)

        # Fallback
        if shape.contour is not None:
            x, y, w, h = cv2.boundingRect(shape.contour)
            return (x + w + 10, y)
        return (50, 50)


class ResultRenderer:
    """Renderiza resultados matemáticos no canvas com estilo JARVIS."""

    @staticmethod
    def draw_math_result(canvas: np.ndarray, result_text: str,
                          position: tuple) -> np.ndarray:
        """
        Desenha o resultado de uma expressão matemática ao lado do '='.
        """
        x, y = position
        text = f" {result_text}"

        # Sombra
        cv2.putText(canvas, text, (x + 2, y + 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        # Texto principal (amarelo-ciano brilhante)
        cv2.putText(canvas, text, (x, y),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 200), 2,
                    cv2.LINE_AA)
        return canvas

    @staticmethod
    def draw_hud_panel(frame: np.ndarray, mode: str, gesture: str,
                        fps: float, width: int, height: int) -> np.ndarray:
        """
        Desenha o painel HUD estilo JARVIS no canto superior.
        """
        # Barra superior
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 70), (5, 10, 20), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Linha divisória
        cv2.line(frame, (0, 70), (width, 70), (0, 180, 180), 1)

        # Título
        cv2.putText(frame, "J.A.R.V.I.S", (20, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 200), 2,
                    cv2.LINE_AA)

        # Modo atual
        mode_text = f"MODO: {mode.upper()}"
        cv2.putText(frame, mode_text, (220, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1,
                    cv2.LINE_AA)

        # Gesto
        if gesture:
            cv2.putText(frame, f"GESTO: {gesture}", (220, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1,
                        cv2.LINE_AA)

        # FPS
        fps_text = f"FPS: {fps:.0f}"
        cv2.putText(frame, fps_text, (width - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 180), 1,
                    cv2.LINE_AA)

        # Legenda de gestos (canto inferior esquerdo)
        legend = [
            "[ 1 dedo ] Desenhar",
            "[ 2 dedos ] Apagar",
            "[ mao aberta ] Analisar",
            "[ pinca ] Mover objeto",
            "[ punho ] Limpar tudo",
            "[ Z ] Desfazer",
            "[ Q ] Sair",
        ]
        for i, item in enumerate(legend):
            y_pos = height - 20 - (len(legend) - 1 - i) * 22
            cv2.putText(frame, item, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 140), 1,
                        cv2.LINE_AA)

        return frame
