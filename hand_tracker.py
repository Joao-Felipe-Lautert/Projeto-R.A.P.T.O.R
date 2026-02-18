"""
JARVIS - Módulo de Rastreamento de Mão (Versão Task API)
Utiliza mediapipe.tasks para maior compatibilidade com Python 3.12+.
"""

import cv2
import mediapipe as mp
import numpy as np
import os

# Tenta importar a nova API de tarefas do MediaPipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    """Rastreia a mão e fornece posição dos dedos via MediaPipe Task API."""

    def __init__(self, max_hands: int = 1, detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.7):
        
        # O MediaPipe Task API requer um arquivo de modelo (.task)
        # Se não existir, tentaremos usar o método clássico com tratamento de erro
        self.model_path = 'hand_landmarker.task'
        self.hand_detected = False
        self.landmarks = []
        self.handedness = []
        
        # Tenta inicializar o detector de tarefas (mais moderno)
        try:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=detection_confidence,
                min_hand_presence_confidence=tracking_confidence,
                min_tracking_confidence=tracking_confidence
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self.use_task_api = True
            print("[INFO] Usando MediaPipe Task API.")
        except Exception as e:
            # Fallback para o método clássico com tratamento de erro robusto
            print(f"[AVISO] Task API falhou: {e}. Tentando método clássico...")
            self.use_task_api = False
            try:
                # Importação tardia para evitar erro no topo do arquivo
                import mediapipe.python.solutions.hands as mp_hands
                import mediapipe.python.solutions.drawing_utils as mp_draw
                self.mp_hands = mp_hands
                self.mp_draw = mp_draw
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=max_hands,
                    min_detection_confidence=detection_confidence,
                    min_tracking_confidence=tracking_confidence
                )
            except Exception as e2:
                print(f"[ERRO CRÍTICO] Falha total no MediaPipe: {e2}")
                self.hands = None

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Processa o frame e extrai landmarks."""
        h, w, _ = frame.shape
        self.landmarks = []
        self.hand_detected = False

        if self.use_task_api:
            # Converte frame para formato MediaPipe
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            # Requer timestamp em milissegundos
            import time
            timestamp_ms = int(time.time() * 1000)
            
            result = self.detector.detect_for_video(mp_image, timestamp_ms)
            
            if result.hand_landmarks:
                self.hand_detected = True
                for hand_lms in result.hand_landmarks:
                    lm_list = []
                    for lm in hand_lms:
                        lm_list.append((int(lm.x * w), int(lm.y * h)))
                    self.landmarks.append(lm_list)
                    
                    # Desenha manualmente para evitar dependência de mp.solutions
                    self._draw_landmarks_manual(frame, lm_list)
        
        elif self.hands:
            # Método clássico
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            if results.multi_hand_landmarks:
                self.hand_detected = True
                for hand_lms in results.multi_hand_landmarks:
                    lm_list = []
                    for lm in hand_lms.landmark:
                        lm_list.append((int(lm.x * w), int(lm.y * h)))
                    self.landmarks.append(lm_list)
                    self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

        return frame

    def _draw_landmarks_manual(self, frame, lm_list):
        """Desenha pontos e conexões manualmente sem mp.solutions."""
        # Conexões simplificadas (dedos)
        connections = [
            (0,1), (1,2), (2,3), (3,4), # polegar
            (0,5), (5,6), (6,7), (7,8), # indicador
            (0,9), (9,10), (10,11), (11,12), # médio
            (0,13), (13,14), (14,15), (15,16), # anelar
            (0,17), (17,18), (18,19), (19,20), # mínimo
            (5,9), (9,13), (13,17) # palma
        ]
        for p1, p2 in connections:
            cv2.line(frame, lm_list[p1], lm_list[p2], (0, 255, 200), 2)
        for pt in lm_list:
            cv2.circle(frame, pt, 4, (0, 200, 255), -1)

    def get_index_finger_tip(self, hand_idx: int = 0):
        if self.landmarks and hand_idx < len(self.landmarks):
            return self.landmarks[hand_idx][8]
        return None

    def fingers_up(self, hand_idx: int = 0) -> list:
        """Retorna lista de booleanos [polegar, indicador, médio, anelar, mínimo]."""
        if not self.landmarks or hand_idx >= len(self.landmarks):
            return [False] * 5

        lm = self.landmarks[hand_idx]
        fingers = []
        
        # Polegar (lógica simples baseada na posição X relativa à base)
        # Ajustado para mão espelhada
        fingers.append(lm[4][0] < lm[3][0])

        # Outros 4 dedos (ponta acima da articulação anterior)
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        for tip, pip in zip(tips, pips):
            fingers.append(lm[tip][1] < lm[pip][1])

        return fingers

    def is_drawing_gesture(self, hand_idx: int = 0) -> bool:
        f = self.fingers_up(hand_idx)
        return f[1] and not f[2] and not f[3] and not f[4]

    def is_erase_gesture(self, hand_idx: int = 0) -> bool:
        f = self.fingers_up(hand_idx)
        return f[1] and f[2] and not f[3] and not f[4]

    def is_confirm_gesture(self, hand_idx: int = 0) -> bool:
        f = self.fingers_up(hand_idx)
        return all(f[1:])

    def is_fist_gesture(self, hand_idx: int = 0) -> bool:
        f = self.fingers_up(hand_idx)
        return not any(f[1:])

    def close(self):
        if self.use_task_api:
            self.detector.close()
        elif self.hands:
            self.hands.close()
