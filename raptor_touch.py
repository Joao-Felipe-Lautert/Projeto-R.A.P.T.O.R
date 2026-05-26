"""
╔══════════════════════════════════════════════════════════════════════╗
║          R.A.P.T.O.R. — Sistema de Desenho Geométrico com Touch     ║
║                    Versão Desktop CORRIGIDA                          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import time
import threading
from raptor_config import BRUSH_SIZE, ERASER_RADIUS, MAX_HISTORY_STEPS
from shape_recognizer import ShapeRecognizer


# Paleta de cores
COLOR_DARK = (52, 21, 11)      # 0, 15, 52
COLOR_PURPLE = (161, 22, 64)   # 64, 22, 161
COLOR_CYAN = (244, 200, 16)    # 16, 200, 244
COLOR_NAVY = (83, 21, 11)      # 11, 21, 83


class RaptorTouchApp:
    """Aplicação RAPTOR com interface touch corrigida."""

    def __init__(self):
        print("╔══════════════════════════════════════════╗")
        print("║  Iniciando R.A.P.T.O.R. Touch ...        ║")
        print("╚══════════════════════════════════════════╝")

        # Resolução
        self.total_width = 1920
        self.total_height = 1200
        self.ui_panel_height = 140
        self.canvas_width = 1920
        self.canvas_height = self.total_height - self.ui_panel_height

        # Canvas
        self.canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        self.canvas[:] = COLOR_DARK
        self.canvas_backup = self.canvas.copy()

        # Estado do desenho
        self.mode = "draw"
        self.is_drawing = False
        self.last_point = None
        self.brush_color = COLOR_CYAN
        self.current_color_index = 0

        # Cores
        self.rgb_colors = [
            COLOR_CYAN,                # Ciano
            COLOR_PURPLE,              # Roxo
            (255, 255, 0),             # Amarelo
            (0, 255, 0),               # Verde
            (255, 0, 0),               # Azul
            (0, 0, 255),               # Vermelho
            (255, 255, 255),           # Branco
        ]

        # Histórico
        self.history = []

        # Módulos
        self.recognizer = ShapeRecognizer(self.canvas_width, self.canvas_height)

        # Análise
        self.analyzing = False
        self.last_analysis_text = ""

        # UI
        self.button_height = 40
        self.button_width = 100
        self.buttons = self._create_buttons()

        # FPS
        self.fps = 0.0
        self.frame_count = 0
        self.fps_timer = time.time()

        print(f"[OK] Resolução: {self.total_width}x{self.total_height}")
        print("[OK] Sistema pronto")

    def _create_buttons(self):
        """Cria os botões."""
        buttons = {}
        x_offset = 15
        y_offset = self.canvas_height + 15
        button_spacing = self.button_width + 12

        button_configs = [
            ("draw", "Desenhar", self.set_draw_mode),
            ("erase", "Apagar", self.set_erase_mode),
            ("analyze", "Analisar", self.trigger_analysis),
            ("undo", "Desfazer", self.undo),
            ("clear", "Limpar", self.clear),
            ("color", "Cor", self.next_color),
        ]

        for i, (key, label, action) in enumerate(button_configs):
            buttons[key] = {
                "x": x_offset + i * button_spacing,
                "y": y_offset,
                "w": self.button_width,
                "h": self.button_height,
                "label": label,
                "action": action,
            }

        return buttons

    def set_draw_mode(self):
        self.mode = "draw"
        print("[MODO] Desenho")

    def set_erase_mode(self):
        self.mode = "erase"
        print("[MODO] Apagar")

    def next_color(self):
        self.current_color_index = (self.current_color_index + 1) % len(self.rgb_colors)
        self.brush_color = self.rgb_colors[self.current_color_index]
        color_name = ["Ciano", "Roxo", "Amarelo", "Verde", "Azul", "Vermelho", "Branco"][self.current_color_index]
        print(f"[COR] {color_name}")

    def trigger_analysis(self):
        if self.analyzing:
            return
        self.analyzing = True
        print("[ANÁLISE] Iniciando...")
        thread = threading.Thread(target=self._analyze_canvas, daemon=True)
        thread.start()

    def _analyze_canvas(self):
        """Analisa formas desenhadas."""
        try:
            # Verifica se há desenho
            diff = cv2.absdiff(self.canvas, self.canvas_backup)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            pixels_drawn = cv2.countNonZero(gray_diff)
            
            print(f"[DEBUG] Pixels desenhados: {pixels_drawn}")
            
            if pixels_drawn < 500:
                print("[RESULTADO] Canvas vazio!")
                self.last_analysis_text = "Desenhe algo primeiro!"
                self.analyzing = False
                return

            # Reconhece formas
            shapes = self.recognizer.analyze_canvas(self.canvas)

            if shapes:
                print(f"[RESULTADO] {len(shapes)} forma(s)")
                text = ""
                for i, shape in enumerate(shapes, 1):
                    print(f"  {i}. {shape.shape_type}")
                    print(f"     Área: {shape.area:.2f} px²")
                    print(f"     Perímetro: {shape.perimeter:.2f} px")
                    print(f"     {shape.description}")
                    text += f"{i}. {shape.description}\n\n"
                self.last_analysis_text = text
            else:
                print("[RESULTADO] Nenhuma forma")
                self.last_analysis_text = "Nenhuma forma detectada"

        except Exception as e:
            print(f"[ERRO] {e}")
            import traceback
            traceback.print_exc()
            self.last_analysis_text = f"Erro: {str(e)}"

        self.analyzing = False

    def undo(self):
        if self.history:
            self.canvas = self.history.pop()
            print("[UNDO]")

    def clear(self):
        self.save_state()
        self.canvas[:] = COLOR_DARK
        self.canvas_backup = self.canvas.copy()
        self.last_analysis_text = ""
        print("[LIMPAR]")

    def save_state(self):
        self.history.append(self.canvas.copy())
        if len(self.history) > MAX_HISTORY_STEPS:
            self.history.pop(0)

    def on_mouse_event(self, event, x, y, flags, param):
        """Evento do mouse."""
        # Clique em botão
        if event == cv2.EVENT_LBUTTONDOWN and y >= self.canvas_height:
            for button_name, button in self.buttons.items():
                if (button["x"] <= x <= button["x"] + button["w"] and
                    button["y"] <= y <= button["y"] + button["h"]):
                    button["action"]()
                    return

        # Desenho no canvas
        if y >= self.canvas_height:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.save_state()
            self.is_drawing = True
            self.last_point = (x, y)
            print(f"[DESENHO] Iniciado em ({x}, {y})")

        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            if self.last_point is None:
                self.last_point = (x, y)
                return

            if self.mode == "draw":
                # Desenha linha do último ponto para o atual
                cv2.line(self.canvas, self.last_point, (x, y), self.brush_color, BRUSH_SIZE, cv2.LINE_AA)
                self.last_point = (x, y)
            elif self.mode == "erase":
                cv2.circle(self.canvas, (x, y), ERASER_RADIUS, COLOR_DARK, -1)

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.last_point = None
            print(f"[DESENHO] Finalizado")

    def draw_ui(self, display):
        """Desenha interface."""
        # Painel
        overlay = display.copy()
        cv2.rectangle(overlay, (0, self.canvas_height), 
                     (self.canvas_width, self.canvas_height + self.ui_panel_height),
                     COLOR_PURPLE, -1)
        cv2.addWeighted(overlay, 0.95, display, 0.05, 0, display)

        # Linha
        cv2.line(display, (0, self.canvas_height), 
                (self.canvas_width, self.canvas_height),
                COLOR_CYAN, 2)

        # Botões
        for button_name, button in self.buttons.items():
            if self.mode == button_name:
                color = COLOR_CYAN
                thickness = 3
            else:
                color = (200, 150, 100)  # Laranja claro para melhor visibilidade
                thickness = 2

            cv2.rectangle(display, (button["x"], button["y"]),
                         (button["x"] + button["w"], button["y"] + button["h"]),
                         color, thickness, cv2.LINE_AA)

            text_size = cv2.getTextSize(button["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = button["x"] + (button["w"] - text_size[0]) // 2
            text_y = button["y"] + (button["h"] + text_size[1]) // 2
            cv2.putText(display, button["label"], (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # Info
        color_name = ["Ciano", "Roxo", "Amarelo", "Verde", "Azul", "Vermelho", "Branco"][self.current_color_index]
        cv2.putText(display, f"Cor: {color_name}", 
                   (self.canvas_width - 220, self.canvas_height + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.brush_color, 1, cv2.LINE_AA)

        cv2.putText(display, "R.A.P.T.O.R. Touch", (15, 35),
                   cv2.FONT_HERSHEY_DUPLEX, 1.3, COLOR_CYAN, 2, cv2.LINE_AA)

        cv2.putText(display, f"MODO: {self.mode.upper()}", (15, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CYAN, 1, cv2.LINE_AA)

        cv2.putText(display, f"{self.canvas_width}x{self.canvas_height}", (15, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PURPLE, 1, cv2.LINE_AA)

        if self.analyzing:
            cv2.putText(display, "Analisando...", (self.canvas_width - 200, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_CYAN, 2, cv2.LINE_AA)

        cv2.putText(display, f"FPS: {self.fps:.0f}", (self.canvas_width - 120, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CYAN, 1, cv2.LINE_AA)

        # Resultados - Tamanho adequado
        if self.last_analysis_text:
            # Fundo para melhor legibilidade
            cv2.rectangle(display, (self.canvas_width - 480, 110), 
                         (self.canvas_width - 20, self.canvas_height - 30),
                         COLOR_DARK, -1)
            cv2.rectangle(display, (self.canvas_width - 480, 110), 
                         (self.canvas_width - 20, self.canvas_height - 30),
                         COLOR_CYAN, 2)
            
            lines = self.last_analysis_text.split('\n')
            for i, line in enumerate(lines[:10]):
                if line.strip():
                    cv2.putText(display, line, (self.canvas_width - 460, 140 + i * 22),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CYAN, 1, cv2.LINE_AA)

        return display

    def run(self):
        """Loop principal."""
        window_name = "R.A.P.T.O.R. Touch"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.total_width, self.total_height)
        cv2.setMouseCallback(window_name, self.on_mouse_event)

        print("[OK] Pronto! Comece a desenhar")

        while True:
            # Display
            display_full = np.zeros((self.total_height, self.total_width, 3), dtype=np.uint8)
            display_full[:self.canvas_height] = self.canvas.copy()
            display_full = self.draw_ui(display_full)

            cv2.imshow(window_name, display_full)

            # Teclas
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key == ord("z") or key == ord("Z"):
                self.undo()
            elif key == ord("c") or key == ord("C"):
                self.clear()
            elif key == ord("a") or key == ord("A"):
                self.trigger_analysis()
            elif key == ord("d") or key == ord("D"):
                self.set_draw_mode()
            elif key == ord("e") or key == ord("E"):
                self.set_erase_mode()
            elif key == ord("x") or key == ord("X"):
                self.next_color()

            self._update_fps()

        cv2.destroyAllWindows()
        print("[OK] Encerrado")

    def _update_fps(self):
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.fps_timer
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_timer = now


def main():
    app = RaptorTouchApp()
    app.run()


if __name__ == "__main__":
    main()
