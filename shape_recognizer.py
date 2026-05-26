"""
╔══════════════════════════════════════════════════════════════════════╗
║          R.A.P.T.O.R. — Módulo de Reconhecimento de Formas          ║
║                    Versão CORRIGIDA                                  ║
╚══════════════════════════════════════════════════════════════════════╝

Detecta e classifica formas geométricas desenhadas no canvas.
"""

import cv2
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Optional
from raptor_config import (
    pixels_to_cm, pixels_square_to_cm_square,
    calculate_circle_area, calculate_circle_perimeter,
    calculate_rectangle_area, calculate_rectangle_perimeter,
    calculate_triangle_area, calculate_triangle_perimeter,
    MIN_CONTOUR_AREA
)


@dataclass
class ShapeResult:
    """Resultado do reconhecimento de uma forma."""
    shape_type: str
    confidence: float
    area: float
    perimeter: float
    params: dict = field(default_factory=dict)
    description: str = ""
    contour: Optional[np.ndarray] = None


class ShapeRecognizer:
    """Reconhece formas geométricas desenhadas."""

    def __init__(self, canvas_width: int = 1920, canvas_height: int = 1060,
                 pixels_per_cm: float = 37.8):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.px_per_cm = pixels_per_cm

    def analyze_canvas(self, canvas: np.ndarray) -> list[ShapeResult]:
        """Analisa o canvas e retorna formas detectadas."""
        # Converte para escala de cinza
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        
        # Threshold mais agressivo para detectar apenas desenhos
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        
        # Dilata para conectar traços
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        # Encontra contornos
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"[DEBUG] {len(contours)} contornos encontrados")
        
        results = []
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            
            print(f"  Contorno {i}: área={area:.0f}, perímetro={perimeter:.0f}")
            
            # Ignora contornos muito pequenos
            if area < 100:  # Mínimo 100 pixels²
                print(f"    → Ignorado (muito pequeno)")
                continue
            
            # Ignora contornos com perímetro muito pequeno
            if perimeter < 30:
                print(f"    → Ignorado (perímetro pequeno)")
                continue
            
            result = self._classify_contour(cnt, area, perimeter)
            if result:
                print(f"    → Detectado: {result.shape_type}")
                results.append(result)
            else:
                print(f"    → Não classificado")
        
        return results

    def _classify_contour(self, contour: np.ndarray, area: float, perimeter: float) -> Optional[ShapeResult]:
        """Classifica um contorno."""
        if perimeter < 1:
            return None
        
        # Aproxima o contorno
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        
        print(f"      Vértices: {vertices}, Circularity: {(4 * math.pi * area) / (perimeter ** 2):.3f}")
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Circularity
        circularity = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        # Classificação
        
        # Círculo (circularity > 0.7)
        if circularity > 0.7:
            return self._make_circle(contour, area, perimeter, circularity)
        
        # Triângulo (3 vértices)
        if vertices == 3:
            return self._make_triangle(approx, area, perimeter)
        
        # Quadrado / Retângulo (4 vértices)
        if vertices == 4:
            return self._make_rectangle(approx, area, perimeter, aspect_ratio)
        
        # Pentágono (5 vértices)
        if vertices == 5:
            return self._make_polygon(approx, area, perimeter, "Pentágono", 5)
        
        # Hexágono (6 vértices)
        if vertices == 6:
            return self._make_polygon(approx, area, perimeter, "Hexágono", 6)
        
        # Polígono genérico
        if 7 <= vertices <= 20:
            return self._make_polygon(approx, area, perimeter, f"Polígono ({vertices} lados)", vertices)
        
        return None

    def _make_circle(self, contour, area, perimeter, circularity) -> ShapeResult:
        """Cria resultado de círculo."""
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        radius = max(5, int(radius))  # Mínimo 5 pixels
        r_cm = pixels_to_cm(radius)
        area_cm2 = calculate_circle_area(r_cm)
        perim_cm = calculate_circle_perimeter(r_cm)
        
        return ShapeResult(
            shape_type="circle",
            confidence=min(circularity, 1.0),
            area=area,
            perimeter=perimeter,
            params={"center": (int(cx), int(cy)), "radius": radius},
            description=(f"Círculo\n"
                        f"Raio: {r_cm:.2f} cm\n"
                        f"Diâmetro: {r_cm * 2:.2f} cm\n"
                        f"Área: {area_cm2:.2f} cm²\n"
                        f"Circunferência: {perim_cm:.2f} cm"),
            contour=contour,
        )

    def _make_triangle(self, approx, area, perimeter) -> ShapeResult:
        """Cria resultado de triângulo."""
        pts = approx.reshape(3, 2)
        sides = []
        for i in range(3):
            p1 = pts[i]
            p2 = pts[(i + 1) % 3]
            dist = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            sides.append(dist)
        
        a, b, c = [pixels_to_cm(s) for s in sides]
        area_cm2 = calculate_triangle_area(a, b, c)
        perim_cm = calculate_triangle_perimeter(a, b, c)
        
        # Classifica tipo
        sides_sorted = sorted([a, b, c])
        if abs(sides_sorted[0] - sides_sorted[2]) < 0.5:
            tri_type = "Equilátero"
        elif abs(sides_sorted[0] - sides_sorted[1]) < 0.5 or abs(sides_sorted[1] - sides_sorted[2]) < 0.5:
            tri_type = "Isósceles"
        else:
            tri_type = "Escaleno"
        
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
        """Cria resultado de retângulo."""
        pts = approx.reshape(4, 2)
        rect = cv2.minAreaRect(approx)
        (cx, cy), (w, h), angle = rect
        
        w_cm = pixels_to_cm(max(w, h))
        h_cm = pixels_to_cm(min(w, h))
        area_cm2 = calculate_rectangle_area(w_cm, h_cm)
        perim_cm = calculate_rectangle_perimeter(w_cm, h_cm)
        
        # Classifica
        if abs(w - h) < 10:
            shape_name = "Quadrado"
        else:
            shape_name = "Retângulo"
        
        return ShapeResult(
            shape_type="rectangle",
            confidence=0.95,
            area=area,
            perimeter=perimeter,
            params={"center": (int(cx), int(cy)), "width": int(w), "height": int(h), "angle": angle},
            description=(f"{shape_name}\n"
                        f"Largura: {w_cm:.2f} cm\n"
                        f"Altura: {h_cm:.2f} cm\n"
                        f"Área: {area_cm2:.2f} cm²\n"
                        f"Perímetro: {perim_cm:.2f} cm"),
            contour=approx,
        )

    def _make_polygon(self, approx, area, perimeter, name: str, sides: int) -> ShapeResult:
        """Cria resultado de polígono."""
        pts = approx.reshape(-1, 2)
        
        # Calcula lados em cm
        side_lengths = []
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            dist = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            side_lengths.append(pixels_to_cm(dist))
        
        perim_cm = sum(side_lengths)
        area_cm2 = pixels_square_to_cm_square(area)
        
        sides_str = ", ".join([f"{s:.2f}" for s in side_lengths])
        
        return ShapeResult(
            shape_type="polygon",
            confidence=0.85,
            area=area,
            perimeter=perimeter,
            params={"points": pts.tolist()},
            description=(f"{name}\n"
                        f"Lados: {sides_str} cm\n"
                        f"Área: {area_cm2:.2f} cm²\n"
                        f"Perímetro: {perim_cm:.2f} cm"),
            contour=approx,
        )
