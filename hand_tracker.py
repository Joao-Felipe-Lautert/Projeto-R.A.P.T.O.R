"""
JARVIS - Módulo de Rastreamento de Mão (Versão Corrigida)
Inclui: Gesto de Pinça (Desenhar), Borracha e Confirmação.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
import time
import math

# Importação da nova API de tarefas do MediaPipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    """Rastreia a mão e fornece posição dos dedos via MediaPipe Task API."""

    def __init__(self, max_hands: int = 1, detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.7):
        
        self.model_path = 'hand_landmarker.task'
        self._check_model()
        
        self.hand_detected = False
        self.landmarks = []
        
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
            print("[OK] MediaPipe Task API inicializada com sucesso.")
        except Exception as e:
            print(f"[ERRO CRÍTICO] Falha ao inicializar MediaPipe: {e}")
            self.detector = None

    def _check_model(self):
        """Verifica se o arquivo de modelo existe, caso contrário, baixa-o."""
        if not os.path.exists(self.model_path):
            print(f"[INFO] Modelo '{self.model_path}' não encontrado. Baixando...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            try:
                urllib.request.urlretrieve(url, self.model_path)
                print("[OK] Modelo baixado com sucesso.")
            except Exception as e:
                print(f"[ERRO] Falha ao baixar o modelo: {e}")

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Processa o frame e extrai landmarks."""
        if self.detector is None:
            return frame

        h, w, _ = frame.shape
        self.landmarks = []
        self.hand_detected = False

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        timestamp_ms = int(time.time() * 1000)
        
        try:
            result = self.detector.detect_for_video(mp_image, timestamp_ms)
            
            if result.hand_landmarks:
                self.hand_detected = True
                for hand_lms in result.hand_landmarks:
                    lm_list = []
                    for lm in hand_lms:
                        lm_list.append((int(lm.x * w), int(lm.y * h)))
                    self.landmarks.append(lm_list)
                    self._draw_landmarks_manual(frame, lm_list)
        except Exception as e:
            print(f"[AVISO] Erro no processamento do frame: {e}")

        return frame

    def _draw_landmarks_manual(self, frame, lm_list):
        """Desenha pontos e conexões manualmente."""
        connections = [
            (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
            (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
            (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17), (0,5), (0,17)
        ]
        for p1, p2 in connections:
            cv2.line(frame, lm_list[p1], lm_list[p2], (200, 255, 0), 2, cv2.LINE_AA)
        for pt in lm_list:
            cv2.circle(frame, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)

    def get_distance(self, p1_idx: int, p2_idx: int, hand_idx: int = 0):
        """Calcula a distância em pixels entre dois pontos."""
        if self.landmarks and hand_idx < len(self.landmarks):
            p1 = self.landmarks[hand_idx][p1_idx]
            p2 = self.landmarks[hand_idx][p2_idx]
            return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        return float('inf')

    def get_index_finger_tip(self, hand_idx: int = 0):
        if self.landmarks and hand_idx < len(self.landmarks):
            return self.landmarks[hand_idx][8]
        return None

    def fingers_up(self, hand_idx: int = 0) -> list:
        """Retorna lista de booleanos para cada dedo levantado."""
        if not self.landmarks or hand_idx >= len(self.landmarks):
            return [False] * 5
        lm = self.landmarks[hand_idx]
        fingers = []
        # Polegar
        fingers.append(lm[4][0] < lm[3][0])
        # Outros 4 dedos (indicador, médio, anelar, mínimo)
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        for tip, pip in zip(tips, pips):
            fingers.append(lm[tip][1] < lm[pip][1])
        return fingers

    def is_drawing_gesture(self, hand_idx: int = 0) -> bool:
        """Ativa quando o polegar (4) e o indicador (8) estão próximos (Pinça)."""
        dist = self.get_distance(4, 8, hand_idx)
        return dist < 50  # Ajuste este valor conforme necessário

    def is_erase_gesture(self, hand_idx: int = 0) -> bool:
        """Borracha: Indicador e Médio levantados."""
        f = self.fingers_up(hand_idx)
        return f[1] and f[2] and not f[3] and not f[4]

    def is_confirm_gesture(self, hand_idx: int = 0) -> bool:
        """Confirmação: Mão aberta (todos os dedos exceto polegar levantados)."""
        f = self.fingers_up(hand_idx)
        return all(f[1:]) # Verifica se indicador, médio, anelar e mínimo estão Up

    def close(self):
        if self.detector:
            self.detector.close()