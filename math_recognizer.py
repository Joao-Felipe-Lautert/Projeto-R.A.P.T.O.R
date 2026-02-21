"""
RAPTOR - Módulo de Reconhecimento Matemático
Reconhece expressões matemáticas desenhadas no canvas usando OCR (Tesseract)
e avalia o resultado com sympy/eval seguro.
"""

import re
import math
import subprocess
import tempfile
import os
import cv2
import numpy as np

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from sympy import sympify, N, pi, sqrt, Rational
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, \
        implicit_multiplication_application
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


# Mapeamento de símbolos OCR para operadores Python
SYMBOL_MAP = {
    "×": "*", "÷": "/", "−": "-", "–": "-",
    "²": "**2", "³": "**3", "√": "sqrt(",
    "π": "pi", "∞": "oo",
    "x": "*",  # comum em OCR confundir × com x
}

# Padrão para detectar expressão matemática com '='
MATH_EXPR_PATTERN = re.compile(
    r"([\d\s\+\-\*\/\(\)\.\^²³√πx×÷−]+)\s*=\s*$"
)


class MathRecognizer:
    """Reconhece e avalia expressões matemáticas desenhadas."""

    def __init__(self):
        self._check_tesseract()

    def _check_tesseract(self):
        """Verifica se Tesseract está disponível."""
        try:
            result = subprocess.run(["tesseract", "--version"],
                                    capture_output=True, text=True)
            self.tesseract_ok = result.returncode == 0
        except FileNotFoundError:
            self.tesseract_ok = False

    def extract_text_from_canvas(self, canvas: np.ndarray) -> str:
        """
        Extrai texto do canvas usando Tesseract OCR.
        Retorna string com o texto reconhecido.
        """
        if not self.tesseract_ok:
            return ""

        # Pré-processa imagem para melhor OCR
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        # Inverte (texto branco → preto para Tesseract)
        inv = cv2.bitwise_not(gray)
        # Aumenta contraste
        _, thresh = cv2.threshold(inv, 30, 255, cv2.THRESH_BINARY)
        # Redimensiona para melhor OCR
        scale = 2.0
        resized = cv2.resize(thresh, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, resized)

        try:
            config = "--psm 6 -c tessedit_char_whitelist=0123456789+-*/=().^"
            text = pytesseract.image_to_string(tmp_path, config=config)
        except Exception:
            text = ""
        finally:
            os.unlink(tmp_path)

        return text.strip()

    def find_math_expression(self, text: str) -> tuple[str, bool]:
        """
        Procura expressão matemática no texto.
        Retorna (expressão, tem_igual).
        """
        # Remove espaços extras
        text = text.strip().replace("\n", " ")

        # Verifica se termina com '='
        has_equal = "=" in text

        # Limpa a expressão
        expr = text.replace("=", "").strip()

        return expr, has_equal

    def evaluate_expression(self, expr: str) -> tuple[str, bool]:
        """
        Avalia uma expressão matemática.
        Retorna (resultado_str, sucesso).
        """
        if not expr:
            return "", False

        # Normaliza símbolos
        clean = self._normalize_expression(expr)

        # Tenta com sympy primeiro
        if SYMPY_AVAILABLE:
            result = self._eval_sympy(clean)
            if result is not None:
                return result, True

        # Fallback: eval seguro
        result = self._safe_eval(clean)
        if result is not None:
            return result, True

        return "", False

    def _normalize_expression(self, expr: str) -> str:
        """Normaliza a expressão para Python."""
        result = expr
        for old, new in SYMBOL_MAP.items():
            result = result.replace(old, new)

        # Fecha parênteses de sqrt se necessário
        result = result.replace("sqrt(", "sqrt(")

        # Remove caracteres inválidos
        result = re.sub(r"[^0-9\+\-\*\/\(\)\.\^sqrt pi]", "", result)

        # Substitui ^ por **
        result = result.replace("^", "**")

        return result.strip()

    def _eval_sympy(self, expr: str) -> str | None:
        """Avalia com sympy."""
        try:
            transformations = standard_transformations + \
                              (implicit_multiplication_application,)
            result = parse_expr(expr, transformations=transformations)
            numeric = float(N(result, 6))

            # Formata resultado
            if numeric == int(numeric):
                return str(int(numeric))
            else:
                return f"{numeric:.4f}".rstrip("0").rstrip(".")
        except Exception:
            return None

    def _safe_eval(self, expr: str) -> str | None:
        """Avalia expressão com eval seguro."""
        safe_globals = {
            "__builtins__": {},
            "sqrt": math.sqrt,
            "pi": math.pi,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "abs": abs,
            "round": round,
            "pow": pow,
        }
        try:
            result = eval(expr, safe_globals)
            if isinstance(result, float):
                if result == int(result):
                    return str(int(result))
                return f"{result:.4f}".rstrip("0").rstrip(".")
            return str(result)
        except Exception:
            return None

    def find_equal_sign_position(self, canvas: np.ndarray) -> tuple | None:
        """
        Detecta a posição do sinal de '=' no canvas para colocar o resultado
        logo ao lado.
        """
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Procura padrão de duas linhas horizontais paralelas (=)
        # Usa morfologia para detectar linhas horizontais
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 2))
        horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)

        contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        equal_candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 15 < w < 80 and h < 15:
                equal_candidates.append((x, y, w, h))

        if len(equal_candidates) >= 2:
            # Agrupa candidatos próximos verticalmente
            equal_candidates.sort(key=lambda c: c[1])
            for i in range(len(equal_candidates) - 1):
                c1 = equal_candidates[i]
                c2 = equal_candidates[i + 1]
                dy = abs(c2[1] - c1[1])
                dx = abs(c2[0] - c1[0])
                if 5 < dy < 25 and dx < 20:
                    # Encontrou o '='
                    x = max(c1[0], c2[0]) + max(c1[2], c2[2]) + 10
                    y = (c1[1] + c2[1]) // 2
                    return (x, y)

        return None

    def process_canvas_for_math(self, canvas: np.ndarray) -> dict:
        """
        Pipeline completo: extrai texto → avalia → retorna resultado.
        """
        result = {
            "text": "",
            "expression": "",
            "value": "",
            "success": False,
            "position": None,
        }

        text = self.extract_text_from_canvas(canvas)
        result["text"] = text

        if not text:
            return result

        expr, has_equal = self.find_math_expression(text)
        result["expression"] = expr

        if expr:
            value, success = self.evaluate_expression(expr)
            result["value"] = value
            result["success"] = success

            if success:
                pos = self.find_equal_sign_position(canvas)
                result["position"] = pos

        return result
