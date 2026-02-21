"""
RAPTOR - Módulo de Assistente de IA
Responde perguntas por digitação usando OpenAI GPT.
Exibe respostas no painel lateral da interface.
"""

import os
import re
import math
import time
import textwrap
from datetime import datetime

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

import cv2
import numpy as np


SYSTEM_PROMPT = """Você é R.A.P.T.O.R. (Just A Rather Very Intelligent System), 
o assistente de IA do Tony Stark. Você é extremamente inteligente, levemente 
sarcástico, mas sempre útil. Responda de forma concisa e direta (máximo 3-4 linhas).
Você pode responder perguntas gerais, fazer cálculos, explicar conceitos científicos,
e dar informações sobre o mundo. Responda sempre em português brasileiro.
Use termos técnicos quando apropriado, mas seja claro."""


# Cores do painel de chat
COLOR_BG       = (8, 15, 25)
COLOR_BORDER   = (0, 160, 160)
COLOR_USER     = (200, 200, 255)
COLOR_RAPTOR   = (0, 255, 200)
COLOR_TITLE    = (0, 220, 255)
COLOR_INPUT    = (255, 255, 200)
COLOR_THINKING = (100, 100, 200)


class AIAssistant:
    """Assistente de IA com interface de chat integrada."""

    def __init__(self, panel_width: int = 400, panel_height: int = 720):
        self.panel_width = panel_width
        self.panel_height = panel_height

        # Histórico de mensagens
        self.messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.chat_history: list[dict] = []  # {role, text, time}

        # Estado do input
        self.input_text: str = ""
        self.thinking: bool = False
        self.last_response: str = ""
        self.is_focused: bool = False  # Indica se o chat está em modo de digitação

        # Cliente OpenAI
        self.client = None
        self._init_client()

        # Scroll
        self.scroll_offset: int = 0

    def _init_client(self):
        """Inicializa o cliente OpenAI."""
        if not OPENAI_AVAILABLE:
            return
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            self.client = OpenAI()

    def add_user_message(self, text: str):
        """Adiciona mensagem do usuário e obtém resposta."""
        if not text.strip():
            return

        # Adiciona ao histórico visual
        self.chat_history.append({
            "role": "user",
            "text": text,
            "time": datetime.now().strftime("%H:%M"),
        })

        # Adiciona ao contexto da IA
        self.messages.append({"role": "user", "content": text})
        self.input_text = ""
        self.thinking = True

        # Obtém resposta
        response = self._get_response(text)
        self.thinking = False

        if response:
            self.messages.append({"role": "assistant", "content": response})
            self.chat_history.append({
                "role": "assistant",
                "text": response,
                "time": datetime.now().strftime("%H:%M"),
            })
            self.last_response = response

        # Auto-scroll para o final
        self.scroll_offset = max(0, len(self.chat_history) - 8)

    def _get_response(self, text: str) -> str:
        """Obtém resposta da IA."""
        # Primeiro tenta resolver matematicamente se for uma expressão
        math_result = self._try_math(text)
        if math_result:
            return math_result

        # Usa OpenAI se disponível
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=self.messages,
                    max_tokens=200,
                    temperature=0.7,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"Erro de conexão: {str(e)[:50]}"

        # Fallback: respostas locais
        return self._local_response(text)

    def _try_math(self, text: str) -> str | None:
        """Tenta resolver expressão matemática simples."""
        # Detecta se é uma pergunta matemática direta
        math_patterns = [
            r"^[\d\s\+\-\*\/\(\)\.\^]+$",
            r"quanto é\s+(.+)",
            r"calcul[ae]\s+(.+)",
            r"(.+)\s*=\s*\?",
        ]

        for pattern in math_patterns:
            match = re.search(pattern, text.lower())
            if match:
                expr = match.group(1) if match.lastindex else text
                expr = re.sub(r"[^0-9\+\-\*\/\(\)\.\^]", "", expr)
                if expr:
                    try:
                        safe_globals = {
                            "__builtins__": {},
                            "sqrt": math.sqrt,
                            "pi": math.pi,
                            "sin": math.sin,
                            "cos": math.cos,
                        }
                        result = eval(expr.replace("^", "**"), safe_globals)
                        return f"O resultado é: {result}"
                    except Exception:
                        pass
        return None

    def _local_response(self, text: str) -> str:
        """Respostas locais sem API (fallback)."""
        text_lower = text.lower()

        if any(w in text_lower for w in ["olá", "oi", "hello", "hey"]):
            return "Olá! Sou R.A.P.T.O.R. Como posso ajudá-lo hoje?"

        if any(w in text_lower for w in ["hora", "horas", "que horas"]):
            return f"São {datetime.now().strftime('%H:%M:%S')}."

        if any(w in text_lower for w in ["data", "hoje", "dia"]):
            return f"Hoje é {datetime.now().strftime('%d/%m/%Y')}."

        if any(w in text_lower for w in ["nome", "quem é você", "quem és"]):
            return "Sou R.A.P.T.O.R. — Just A Rather Very Intelligent System."

        if any(w in text_lower for w in ["obrigado", "valeu", "thanks"]):
            return "Sempre às ordens, senhor."

        if "pi" in text_lower:
            return f"π ≈ {math.pi:.10f}"

        if "euler" in text_lower or " e " in text_lower:
            return f"Número de Euler: e ≈ {math.e:.10f}"

        return ("Não tenho conexão com a API no momento. "
                "Configure OPENAI_API_KEY para respostas completas.")

    # ------------------------------------------------------------------ #
    #  Renderização do painel de chat                                      #
    # ------------------------------------------------------------------ #

    def render_panel(self, frame: np.ndarray,
                      x_offset: int = 0) -> np.ndarray:
        """
        Renderiza o painel de chat no lado direito do frame.
        """
        h, w = frame.shape[:2]
        pw = self.panel_width

        # Fundo do painel
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_offset, 0), (x_offset + pw, h),
                      COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)

        # Borda esquerda
        cv2.line(frame, (x_offset, 0), (x_offset, h), COLOR_BORDER, 2)

        # Título
        cv2.putText(frame, "[ R.A.P.T.O.R. CHAT ]",
                    (x_offset + 10, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, COLOR_TITLE, 1,
                    cv2.LINE_AA)
        cv2.line(frame, (x_offset, 55), (x_offset + pw, 55), COLOR_BORDER, 1)

        # Área de mensagens
        self._render_messages(frame, x_offset, 65, h - 100)

        # Área de input
        self._render_input_box(frame, x_offset, h - 90, pw)

        return frame

    def _render_messages(self, frame, x, y_start, y_end):
        """Renderiza as mensagens do chat."""
        pw = self.panel_width - 20
        y = y_start + 10
        line_h = 20

        visible = self.chat_history[self.scroll_offset:]

        for msg in visible:
            if y >= y_end:
                break

            is_user = msg["role"] == "user"
            color = COLOR_USER if is_user else COLOR_RAPTOR
            prefix = "Você: " if is_user else "RAPTOR: "
            font_scale = 0.45
            thickness = 1

            # Quebra texto em linhas
            full_text = prefix + msg["text"]
            max_chars = (pw - 10) // 8
            lines = textwrap.wrap(full_text, width=max(max_chars, 20))

            for line in lines[:4]:  # máximo 4 linhas por mensagem
                if y >= y_end:
                    break
                cv2.putText(frame, line, (x + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                            thickness, cv2.LINE_AA)
                y += line_h

            # Timestamp
            if y < y_end:
                cv2.putText(frame, msg["time"], (x + pw - 40, y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 100),
                            1, cv2.LINE_AA)

            # Separador
            y += 5
            if y < y_end:
                cv2.line(frame, (x + 10, y), (x + pw, y), (20, 30, 40), 1)
            y += 8

        # Indicador "pensando..."
        if self.thinking:
            cv2.putText(frame, "RAPTOR processando...", (x + 10, y_end - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_THINKING, 1,
                        cv2.LINE_AA)

    def _render_input_box(self, frame, x, y, pw):
        """Renderiza a caixa de input."""
        # Cor da borda muda se estiver focado
        border_color = (0, 255, 200) if self.is_focused else COLOR_BORDER
        bg_color = (25, 35, 45) if self.is_focused else (15, 25, 35)

        # Fundo
        cv2.rectangle(frame, (x + 5, y), (x + pw - 5, y + 70),
                      bg_color, -1)
        cv2.rectangle(frame, (x + 5, y), (x + pw - 5, y + 70),
                      border_color, 2 if self.is_focused else 1)

        # Label
        label = "[ MODO DIGITACAO ]" if self.is_focused else "Pressione TAB para digitar:"
        cv2.putText(frame, label,
                    (x + 10, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200) if self.is_focused else (0, 140, 140), 1,
                    cv2.LINE_AA)

        # Texto digitado
        cursor = "_" if (int(time.time() * 2) % 2 == 0 and self.is_focused) else ""
        display_text = self.input_text[-45:] + cursor
        cv2.putText(frame, display_text, (x + 10, y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_INPUT, 1,
                    cv2.LINE_AA)

    def handle_key(self, key: int) -> bool:
        """
        Processa tecla pressionada.
        Retorna True se deve enviar mensagem.
        """
        # Se pressionar TAB, alterna o foco do chat
        if key == 9:  # TAB
            self.is_focused = not self.is_focused
            return False

        if not self.is_focused:
            return False

        if key == 13:  # Enter
            if self.input_text.strip():
                self.add_user_message(self.input_text)
                self.is_focused = False # Perde o foco após enviar
                return True
        elif key == 8:  # Backspace
            self.input_text = self.input_text[:-1]
        elif key == 27:  # Esc - limpa input ou sai do foco
            if self.input_text:
                self.input_text = ""
            else:
                self.is_focused = False
        elif 32 <= key <= 126:  # Caracteres imprimíveis
            if len(self.input_text) < 200:
                self.input_text += chr(key)
        return False
