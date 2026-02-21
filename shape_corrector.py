"""
RAPTOR - Módulo de Correção de Formas
Redesenha formas tortas de maneira geométrica perfeita sobre o canvas.
"""

import cv2
import numpy as np
import math
from shape_recognizer import ShapeResult
import mediapipe as mp


# Cores estilo RAPTOR
COLOR_CORRECTED = (0, 255, 180)    # Verde-ciano neon
COLOR_ORIGINAL  = (40, 40, 40)     # Cinza escuro (apaga original)
COLOR_LABEL     = (0, 220, 255)    # Amarelo-ciano
COLOR_MEASURE   = (180, 255, 180)  # Verde claro

# --- Constantes de Estilo ---
COLOR_CORRECTED = (0, 255, 255) # Ciano futurista
COLOR_LABEL = (255, 255, 255)
COLOR_MEASURE = (0, 255, 0)

class ShapeCorrector:
    def __init__(self, canvas_width: int = 1280, canvas_height: int = 720):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

    def correct_and_draw(self, canvas: np.ndarray, shape, erase_original: bool = True) -> np.ndarray:
        # Criamos uma cópia para não destruir o frame original permanentemente
        result = canvas.copy()

        if erase_original and hasattr(shape, 'contour') and shape.contour is not None:
            self._erase_original(result, shape.contour)

        self._draw_corrected(result, shape)
        self._draw_measurements(result, shape)
        return result

    def _erase_original(self, canvas: np.ndarray, contour: np.ndarray):
        """Cria o efeito de 'apagar' o rascunho original."""
        mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        # Dilatação para garantir que não sobrem 'sujeiras' do rascunho
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.dilate(mask, kernel)
        canvas[mask > 0] = [0, 0, 0]

    def _draw_corrected(self, canvas: np.ndarray, shape):
        t = 3
        color = COLOR_CORRECTED
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

        # ... (outras formas: triangle, line, etc)

    def _draw_corner_marks(self, canvas, x, y, w, h, color, size=12):
        corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        for (cx, cy), (dx, dy) in zip(corners, dirs):
            cv2.line(canvas, (cx, cy), (cx + dx * size, cy), color, 2, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy), (cx, cy + dy * size), color, 2, cv2.LINE_AA)

    def _draw_measurements(self, canvas: np.ndarray, shape):
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
        
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (x, y + i * 22), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_LABEL, 1, cv2.LINE_AA)

    def _get_label_position(self, shape):
        if shape.shape_type == "circle":
            cx, cy = shape.params["center"]
            return (cx + shape.params["radius"] + 15, cy)
        elif shape.shape_type == "rectangle":
            return (shape.params["x"] + shape.params["w"] + 15, shape.params["y"])
        return (50, 50)

# --- Lógica de Controle de Estado ---

def main():
    cap = cv2.VideoCapture(0)
    corrector = ShapeCorrector()
    
    # MediaPipe Setup
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)

    # Estado do Sistema
    analise_bloqueada = False
    resultado_fixado = None 

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        mao_aberta = True
        if results.multi_hand_landmarks:
            # Lógica simples: se o dedo indicador estiver abaixo do nó do dedo, está fechada
            lm = results.multi_hand_landmarks[0].landmark
            mao_aberta = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[mp_hands.HandLandmark.INDEX_FINGER_MCP].y

        # --- MÁQUINA DE ESTADOS ---
        
        if mao_aberta:
            # 1. Reset: Abre a mão -> Limpa a tela e libera nova análise
            analise_bloqueada = False
            resultado_fixado = None
            cv2.putText(frame, "STATUS: PRONTO (Mao Aberta)", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # 2. Gatilho: Fechou a mão pela primeira vez
            if not analise_bloqueada:
                # AQUI VOCÊ CHAMA SEU DETECTOR DE FORMAS
                # resultado_fixado = detector.detect(frame)
                # analise_bloqueada = True
                pass
            
            cv2.putText(frame, "STATUS: TRAVADO (Mao Fechada)", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 3. Renderização persistente
        if resultado_fixado:
            # Redesenha a correção e o "quadrado preto" por cima do frame atual da câmera
            frame = corrector.correct_and_draw(frame, resultado_fixado)

        cv2.imshow("Vision System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

class ResultRenderer:
    """Renderiza resultados matemáticos no canvas com estilo RAPTOR."""

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

# --- HUD Panel ---

    @staticmethod
    def draw_hud_panel(frame: np.ndarray, mode: str, gesture: str,
                        fps: float, width: int, height: int) -> np.ndarray:
        """
        Desenha o painel HUD estilo RAPTOR no canto superior.
        """
        # Barra superior
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 70), (5, 10, 20), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Linha divisória
        cv2.line(frame, (0, 70), (width, 70), (0, 180, 180), 1)

        # Título
        cv2.putText(frame, "R.A.P.T.O.R.", (20, 45),
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
            "[ Pinca ] Desenhar",
            "[ 2 dedos ] Apagar",
            "[ mao aberta ] Analisar",
            "[ C ] Limpar tudo",
            "[ Z ] Desfazer",
            "[ Q ou Esc ] Sair",
        ]
        for i, item in enumerate(legend):
            y_pos = height - 20 - (len(legend) - 1 - i) * 22
            cv2.putText(frame, item, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 140), 1,
                        cv2.LINE_AA)

        return frame
