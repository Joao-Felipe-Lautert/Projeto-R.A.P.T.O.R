"""
RAPTOR - Módulo de Reconhecimento de Formas
Detecta e classifica formas geométricas desenhadas no canvas.
Calcula área, perímetro e corrige formas tortas.
"""

import cv2
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ShapeResult:
    """Resultado do reconhecimento de uma forma."""
    shape_type: str          # "circle", "rectangle", "triangle", "line", "polygon", "unknown"
    confidence: float        # 0.0 a 1.0
    area: float              # pixels²
    perimeter: float         # pixels
    params: dict = field(default_factory=dict)   # parâmetros da forma corrigida
    description: str = ""    # texto descritivo com medidas
    contour: Optional[np.ndarray] = None


class ShapeRecognizer:
    """Reconhece formas geométricas a partir dos pontos desenhados."""

    def __init__(self, canvas_width: int = 1280, canvas_height: int = 720,
                 pixels_per_cm: float = 37.8):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.px_per_cm = pixels_per_cm  # ~96 DPI padrão

    def analyze_canvas(self, canvas: np.ndarray) -> list[ShapeResult]:
        """
        Analisa o canvas e retorna lista de formas detectadas.
        """
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Dilata para conectar traços próximos
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # ignora ruídos pequenos
                continue
            result = self._classify_contour(cnt)
            if result:
                results.append(result)

        return results

    def _classify_contour(self, contour: np.ndarray) -> Optional[ShapeResult]:
        """Classifica um contorno em uma forma geométrica."""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter < 1:
            return None

        # Aproxima o contorno
        epsilon = 0.03 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)

        # Compacidade (circularidade): 4π·A / P²
        circularity = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 1.0

        # --- Classificação ---

        # Círculo / Elipse
        if circularity > 0.82:
            return self._make_circle(contour, area, perimeter, circularity)

        # Linha (muito alongada)
        if vertices == 2 or (vertices <= 4 and circularity < 0.1):
            return self._make_line(contour, area, perimeter)

        # Triângulo
        if vertices == 3:
            return self._make_triangle(approx, area, perimeter)

        # Quadrado / Retângulo
        if vertices == 4:
            return self._make_rectangle(approx, area, perimeter, aspect_ratio)

        # Pentágono
        if vertices == 5:
            return self._make_polygon(approx, area, perimeter, "Pentágono", 5)

        # Hexágono
        if vertices == 6:
            return self._make_polygon(approx, area, perimeter, "Hexágono", 6)

        # Polígono genérico
        if 7 <= vertices <= 12:
            return self._make_polygon(approx, area, perimeter,
                                      f"Polígono ({vertices} lados)", vertices)

        # Forma desconhecida / livre
        return ShapeResult(
            shape_type="unknown",
            confidence=0.3,
            area=area,
            perimeter=perimeter,
            params={"contour": contour},
            description=f"Forma livre\nÁrea: {self._px2cm2(area):.2f} cm²\n"
                        f"Perímetro: {self._px2cm(perimeter):.2f} cm",
            contour=contour,
        )

    # ------------------------------------------------------------------ #
    #  Construtores de formas                                              #
    # ------------------------------------------------------------------ #

    def _make_circle(self, contour, area, perimeter, circularity) -> ShapeResult:
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        cx, cy, radius = int(cx), int(cy), int(radius)
        r_cm = self._px2cm(radius)
        area_cm2 = math.pi * r_cm ** 2
        perim_cm = 2 * math.pi * r_cm

        # Detecta elipse se aspect ratio muito diferente
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (ex, ey), (ma, mi), angle = ellipse
            if abs(ma - mi) / max(ma, mi) > 0.25:
                a_cm = self._px2cm(ma / 2)
                b_cm = self._px2cm(mi / 2)
                area_cm2 = math.pi * a_cm * b_cm
                # Aproximação de Ramanujan para perímetro da elipse
                h = ((a_cm - b_cm) / (a_cm + b_cm)) ** 2
                perim_cm = math.pi * (a_cm + b_cm) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))
                return ShapeResult(
                    shape_type="ellipse",
                    confidence=circularity,
                    area=area,
                    perimeter=perimeter,
                    params={"center": (int(ex), int(ey)),
                            "axes": (int(ma / 2), int(mi / 2)),
                            "angle": angle},
                    description=(f"Elipse\n"
                                 f"Semi-eixo maior: {a_cm:.2f} cm\n"
                                 f"Semi-eixo menor: {b_cm:.2f} cm\n"
                                 f"Área: {area_cm2:.2f} cm²\n"
                                 f"Perímetro ≈ {perim_cm:.2f} cm"),
                    contour=contour,
                )

        return ShapeResult(
            shape_type="circle",
            confidence=min(circularity, 1.0),
            area=area,
            perimeter=perimeter,
            params={"center": (cx, cy), "radius": radius},
            description=(f"Círculo\n"
                         f"Raio: {r_cm:.2f} cm\n"
                         f"Diâmetro: {r_cm * 2:.2f} cm\n"
                         f"Área: {area_cm2:.2f} cm²\n"
                         f"Circunferência: {perim_cm:.2f} cm"),
            contour=contour,
        )

    def _make_line(self, contour, area, perimeter) -> ShapeResult:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        # Pega os dois pontos extremos
        pts = contour.reshape(-1, 2)
        p1 = tuple(pts[0])
        p2 = tuple(pts[-1])
        length = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        length_cm = self._px2cm(length)
        angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

        return ShapeResult(
            shape_type="line",
            confidence=0.8,
            area=area,
            perimeter=perimeter,
            params={"p1": p1, "p2": p2},
            description=(f"Linha\n"
                         f"Comprimento: {length_cm:.2f} cm\n"
                         f"Ângulo: {angle:.1f}°"),
            contour=contour,
        )

    def _make_triangle(self, approx, area, perimeter) -> ShapeResult:
        pts = approx.reshape(3, 2)
        sides = []
        for i in range(3):
            p1 = pts[i]
            p2 = pts[(i + 1) % 3]
            sides.append(math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2))

        a, b, c = [self._px2cm(s) for s in sides]
        # Fórmula de Heron
        s = (a + b + c) / 2
        area_cm2 = math.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
        perim_cm = a + b + c

        # Classifica tipo de triângulo
        sides_sorted = sorted([a, b, c])
        if abs(sides_sorted[0] - sides_sorted[2]) < 0.5:
            tri_type = "Equilátero"
        elif abs(sides_sorted[0] - sides_sorted[1]) < 0.5 or \
             abs(sides_sorted[1] - sides_sorted[2]) < 0.5:
            tri_type = "Isósceles"
        else:
            tri_type = "Escaleno"

        # Verifica se é retângulo
        a2, b2, c2 = sorted([sides[0] ** 2, sides[1] ** 2, sides[2] ** 2])
        if abs(a2 + b2 - c2) / c2 < 0.1:
            tri_type += " Retângulo"

        return ShapeResult(
            shape_type="triangle",
            confidence=0.9,
            area=area,
            perimeter=perimeter,
            params={"points": pts.tolist()},
            description=(f"Triângulo {tri_type}\n"
                         f"Lados: {a:.2f}, {b:.2f}, {c:.2f} cm\n"
                         f"Área: {area_cm2:.2f} cm²\n"
                         f"Perímetro: {perim_cm:.2f} cm"),
            contour=approx,
        )

    def _make_rectangle(self, approx, area, perimeter, aspect_ratio) -> ShapeResult:
        pts = approx.reshape(4, 2)
        rect = cv2.minAreaRect(approx)
        (cx, cy), (w, h), angle = rect

        w_cm = self._px2cm(max(w, h))
        h_cm = self._px2cm(min(w, h))
        area_cm2 = w_cm * h_cm
        perim_cm = 2 * (w_cm + h_cm)

        x, y = int(cx - max(w, h) / 2), int(cy - min(w, h) / 2)
        shape_name = "Quadrado" if abs(aspect_ratio - 1.0) < 0.15 else "Retângulo"

        return ShapeResult(
            shape_type="rectangle",
            confidence=0.9,
            area=area,
            perimeter=perimeter,
            params={"x": x, "y": y, "w": int(max(w, h)), "h": int(min(w, h))},
            description=(f"{shape_name}\n"
                         f"Largura: {w_cm:.2f} cm\n"
                         f"Altura: {h_cm:.2f} cm\n"
                         f"Área: {area_cm2:.2f} cm²\n"
                         f"Perímetro: {perim_cm:.2f} cm"),
            contour=approx,
        )

    def _make_polygon(self, approx, area, perimeter, name, n) -> ShapeResult:
        pts = approx.reshape(n, 2)
        area_cm2 = self._px2cm2(area)
        perim_cm = self._px2cm(perimeter)

        return ShapeResult(
            shape_type="polygon",
            confidence=0.8,
            area=area,
            perimeter=perimeter,
            params={"points": pts.tolist()},
            description=(f"{name}\n"
                         f"Área: {area_cm2:.2f} cm²\n"
                         f"Perímetro: {perim_cm:.2f} cm"),
            contour=approx,
        )

    # ------------------------------------------------------------------ #
    #  Utilitários de conversão                                            #
    # ------------------------------------------------------------------ #

    def _px2cm(self, px: float) -> float:
        return px / self.px_per_cm

    def _px2cm2(self, px2: float) -> float:
        return px2 / (self.px_per_cm ** 2)
