"""
╔══════════════════════════════════════════════════════════════════════╗
║          J.A.R.V.I.S. — Just A Rather Very Intelligent System       ║
║                    Desenvolvido em Python + OpenCV                   ║
╚══════════════════════════════════════════════════════════════════════╝

Funcionalidades:
  • Desenho no ar com o dedo indicador via webcam
  • Reconhecimento e correção de formas geométricas
  • Cálculo automático de área, perímetro e medidas
  • Reconhecimento de expressões matemáticas desenhadas
  • Assistente de IA para responder perguntas por digitação

Gestos:
  ✦ 1 dedo (indicador)     → Desenhar
  ✦ 2 dedos (ind. + médio) → Apagar (borracha)
  ✦ Mão aberta             → Analisar desenho
  ✦ Punho fechado          → Limpar canvas
  ✦ [Z]                    → Desfazer
  ✦ [Q] ou [ESC]           → Sair
"""

import cv2
import numpy as np
import time
import threading
import sys
import os

# Adiciona o diretório atual ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hand_tracker import HandTracker
from canvas import DrawingCanvas, COLORS
from shape_recognizer import ShapeRecognizer
from shape_corrector import ShapeCorrector, ResultRenderer
from math_recognizer import MathRecognizer
from ai_assistant import AIAssistant


# ──────────────────────────────────────────────────────────────────────
#  Configurações
# ──────────────────────────────────────────────────────────────────────

CAM_WIDTH      = 1280
CAM_HEIGHT     = 720
CHAT_PANEL_W   = 380
TOTAL_WIDTH    = CAM_WIDTH + CHAT_PANEL_W

GESTURE_HOLD   = 0.6   # segundos para confirmar gesto de análise
FIST_HOLD      = 0.8   # segundos para confirmar limpeza
ERASE_RADIUS   = 35    # raio da borracha


# ──────────────────────────────────────────────────────────────────────
#  Classe principal
# ──────────────────────────────────────────────────────────────────────

class JARVIS:
    def __init__(self, camera_index: int = 0):
        print("╔══════════════════════════════════════════╗")
        print("║  Iniciando J.A.R.V.I.S. ...              ║")
        print("╚══════════════════════════════════════════╝")

        # Câmera
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            print("[ERRO] Não foi possível abrir a câmera.")
            sys.exit(1)

        # Módulos
        try:
            self.tracker   = HandTracker(detection_confidence=0.75,
                                         tracking_confidence=0.75)
        except Exception as e:
            print(f"[ERRO] Falha ao inicializar HandTracker: {e}")
            self.tracker = None
            
        self.canvas    = DrawingCanvas(CAM_WIDTH, CAM_HEIGHT)
        self.recognizer = ShapeRecognizer(CAM_WIDTH, CAM_HEIGHT)
        self.corrector = ShapeCorrector(CAM_WIDTH, CAM_HEIGHT)
        self.math_rec  = MathRecognizer()
        self.ai        = AIAssistant(CHAT_PANEL_W, CAM_HEIGHT)

        # Estado
        self.mode          = "draw"   # draw | erase | analyze | idle
        self.gesture_label = ""
        self.prev_point    = None
        self.fps           = 0.0
        self.frame_count   = 0
        self.fps_timer     = time.time()

        # Timers de gestos
        self.confirm_start = 0.0
        self.fist_start    = 0.0
        self.confirm_shown = False

        # Análise em thread separada
        self.analyzing     = False
        self.analysis_done = False
        self.analysis_results = []

        # Último ponto para suavização
        self.smooth_points = []
        self.smooth_window = 3

        print("[OK] Todos os módulos carregados.")
        print("[INFO] Pressione Q ou ESC para sair.")

    # ──────────────────────────────────────────────────────────────────
    #  Loop principal
    # ──────────────────────────────────────────────────────────────────

    def run(self):
        cv2.namedWindow("J.A.R.V.I.S.", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("J.A.R.V.I.S.", TOTAL_WIDTH, CAM_HEIGHT)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERRO] Frame inválido da câmera.")
                break

            # Espelha horizontalmente (mais natural)
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))

            # Processa mão
            if self.tracker:
                frame = self.tracker.process(frame)

            # Atualiza gestos e desenho
            self._process_gestures(frame)

            # Compõe canvas + frame
            composed = self.canvas.get_canvas_with_overlay(frame)

            # Desenha HUD
            composed = ResultRenderer.draw_hud_panel(
                composed, self.mode, self.gesture_label,
                self.fps, CAM_WIDTH, CAM_HEIGHT
            )

            # Indicador de análise em andamento
            if self.analyzing:
                self._draw_analyzing_indicator(composed)

            # Painel de chat (lado direito)
            full_frame = np.zeros((CAM_HEIGHT, TOTAL_WIDTH, 3), dtype=np.uint8)
            full_frame[:, :CAM_WIDTH] = composed
            self.ai.render_panel(full_frame, x_offset=CAM_WIDTH)

            # Atualiza FPS
            self._update_fps()

            cv2.imshow("J.A.R.V.I.S.", full_frame)

            # Processa teclas
            key = cv2.waitKey(1) & 0xFF
            
            # Se o chat estiver focado, envia todas as teclas para ele
            if self.ai.is_focused:
                self.ai.handle_key(key)
            else:
                # Atalhos globais (apenas se o chat NÃO estiver focado)
                if key in (ord("q"), ord("Q"), 27):
                    break
                elif key == ord("z") or key == ord("Z"):
                    self.canvas.undo()
                elif key == ord("c") or key == ord("C"):
                    self.canvas.clear()
                elif key == ord("a") or key == ord("A"):
                    self._trigger_analysis()
                elif key == 9: # TAB para focar no chat
                    self.ai.is_focused = True

        self._cleanup()

    # ──────────────────────────────────────────────────────────────────
    #  Processamento de gestos
    # ──────────────────────────────────────────────────────────────────

    def _process_gestures(self, frame: np.ndarray):
        """Processa gestos da mão e atualiza o canvas."""
        if not self.tracker or not self.tracker.hand_detected:
            self.mode = "idle"
            self.gesture_label = "Nenhuma mão"
            if self.canvas.drawing:
                self.canvas.end_stroke()
            self.prev_point = None
            return

        # Obtém ponto do indicador
        index_tip = self.tracker.get_index_finger_tip()
        if index_tip is None:
            return

        # Suaviza o ponto
        smooth_pt = self._smooth_point(index_tip)

        # Detecta gestos
        is_drawing  = self.tracker.is_drawing_gesture()
        is_erasing  = self.tracker.is_erase_gesture()
        is_confirm  = self.tracker.is_confirm_gesture()
        is_fist     = self.tracker.is_fist_gesture()

        now = time.time()

        # ── Punho: limpar canvas ──
        if is_fist:
            if self.fist_start == 0:
                self.fist_start = now
            elif now - self.fist_start > FIST_HOLD:
                self.canvas.clear()
                self.fist_start = 0
                self.gesture_label = "Canvas limpo!"
            else:
                remaining = FIST_HOLD - (now - self.fist_start)
                self.gesture_label = f"Segure para limpar ({remaining:.1f}s)"
            self.mode = "idle"
            if self.canvas.drawing:
                self.canvas.end_stroke()
            return
        else:
            self.fist_start = 0

        # ── Mão aberta: analisar ──
        if is_confirm:
            if self.confirm_start == 0:
                self.confirm_start = now
            elif now - self.confirm_start > GESTURE_HOLD and not self.analyzing:
                self._trigger_analysis()
                self.confirm_start = 0
            else:
                remaining = GESTURE_HOLD - (now - self.confirm_start)
                self.gesture_label = f"Analisando em {remaining:.1f}s..."
            self.mode = "analyze"
            if self.canvas.drawing:
                self.canvas.end_stroke()
            return
        else:
            self.confirm_start = 0

        # ── Dois dedos: borracha ──
        if is_erasing:
            self.mode = "erase"
            self.gesture_label = "Borracha"
            if self.canvas.drawing:
                self.canvas.end_stroke()
            self.canvas.erase_area(smooth_pt, ERASE_RADIUS)
            # Desenha cursor de borracha
            cv2.circle(frame, smooth_pt, ERASE_RADIUS,
                       (100, 100, 100), 2, cv2.LINE_AA)
            self.prev_point = None
            return

        # ── Um dedo: desenhar ──
        if is_drawing:
            self.mode = "draw"
            self.gesture_label = "Desenhando"

            if not self.canvas.drawing:
                self.canvas.save_state()
                self.canvas.start_stroke(smooth_pt)
            else:
                self.canvas.add_point(smooth_pt)

            # Cursor do dedo
            cv2.circle(frame, smooth_pt, self.canvas.brush_size + 2,
                       COLORS["draw"], -1, cv2.LINE_AA)
            self.prev_point = smooth_pt
            return

        # Nenhum gesto específico
        self.mode = "idle"
        self.gesture_label = "Pronto"
        if self.canvas.drawing:
            self.canvas.end_stroke()
        self.prev_point = None

    def _smooth_point(self, point: tuple) -> tuple:
        """Suaviza o ponto do dedo com média móvel."""
        self.smooth_points.append(point)
        if len(self.smooth_points) > self.smooth_window:
            self.smooth_points.pop(0)
        x = int(sum(p[0] for p in self.smooth_points) / len(self.smooth_points))
        y = int(sum(p[1] for p in self.smooth_points) / len(self.smooth_points))
        return (x, y)

    # ──────────────────────────────────────────────────────────────────
    #  Análise do canvas
    # ──────────────────────────────────────────────────────────────────

    def _trigger_analysis(self):
        """Inicia análise do canvas em thread separada."""
        if self.analyzing:
            return
        self.analyzing = True
        self.gesture_label = "Analisando..."
        thread = threading.Thread(target=self._analyze_canvas, daemon=True)
        thread.start()

    def _analyze_canvas(self):
        """Analisa o canvas: detecta formas e expressões matemáticas."""
        canvas_copy = self.canvas.canvas.copy()

        # 1. Tenta reconhecer expressão matemática
        math_result = self.math_rec.process_canvas_for_math(canvas_copy)

        if math_result["success"] and math_result["value"]:
            # Coloca resultado ao lado do '='
            pos = math_result.get("position")
            if pos:
                ResultRenderer.draw_math_result(
                    self.canvas.canvas,
                    math_result["value"],
                    pos
                )
            else:
                # Coloca no centro-direito
                self.canvas.draw_result(
                    f"= {math_result['value']}",
                    (CAM_WIDTH // 2 + 50, CAM_HEIGHT // 2)
                )
            self.analyzing = False
            return

        # 2. Reconhece formas geométricas
        shapes = self.recognizer.analyze_canvas(canvas_copy)

        if shapes:
            # Corrige cada forma detectada
            for shape in shapes:
                self.canvas.canvas = self.corrector.correct_and_draw(
                    self.canvas.canvas, shape, erase_original=True
                )

            # Exibe resumo no chat
            summary = self._build_shape_summary(shapes)
            self.ai.chat_history.append({
                "role": "assistant",
                "text": summary,
                "time": __import__("datetime").datetime.now().strftime("%H:%M"),
            })
        else:
            self.canvas.add_object("text", {"text": "Nenhuma forma detectada.", "pos": (CAM_WIDTH // 4, CAM_HEIGHT // 2)})

        self.analyzing = False

    def _build_shape_summary(self, shapes) -> str:
        """Constrói resumo das formas detectadas."""
        if not shapes:
            return "Nenhuma forma detectada."

        lines = [f"{len(shapes)} forma(s) detectada(s):"]
        for i, s in enumerate(shapes[:3], 1):
            first_line = s.description.split("\n")[0]
            lines.append(f"{i}. {first_line}")
        return "\n".join(lines)

    def _draw_analyzing_indicator(self, frame: np.ndarray):
        """Desenha indicador de análise em andamento."""
        t = time.time()
        dots = "." * (int(t * 3) % 4)
        text = f"Analisando{dots}"
        cv2.putText(frame, text, (CAM_WIDTH // 2 - 80, CAM_HEIGHT // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 200), 2,
                    cv2.LINE_AA)

    # ──────────────────────────────────────────────────────────────────
    #  Utilitários
    # ──────────────────────────────────────────────────────────────────

    def _update_fps(self):
        """Atualiza contador de FPS."""
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.fps_timer
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_timer = now

    def _cleanup(self):
        """Libera recursos."""
        print("\n[INFO] Encerrando J.A.R.V.I.S. ...")
        self.tracker.close()
        self.cap.release()
        cv2.destroyAllWindows()
        print("[OK] Até logo, senhor.")


# ──────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="J.A.R.V.I.S. — Assistente de IA com visão computacional"
    )
    parser.add_argument("--camera", type=int, default=0,
                        help="Índice da câmera (padrão: 0)")
    args = parser.parse_args()

    jarvis = JARVIS(camera_index=args.camera)
    jarvis.run()


if __name__ == "__main__":
    main()
