"""Reusable Vietnamese Sign Language recognition helpers.
UPDATED: Synced with Training Logic (Padding + 33 Pose Landmarks)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

mp_holistic = mp.solutions.holistic

# --- CẤU HÌNH KHỚP VỚI TRAIN.PY ---
# Lúc train ta dùng full 33 điểm pose
N_POSE_LANDMARKS = 33 
N_HAND_LANDMARKS = 21
# Tổng: (33 + 21 + 21) * 3 = 225 features
N_TOTAL_LANDMARKS = (N_POSE_LANDMARKS + 2 * N_HAND_LANDMARKS) * 3
TARGET_SEQUENCE_LENGTH = 60


def create_holistic(
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    """Create a MediaPipe Holistic model."""
    return mp_holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def load_label_map(label_map_path: Path) -> tuple[dict, dict]:
    path = Path(label_map_path)
    with path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    # Đảo ngược: Số -> Chữ
    inv_label_map = {v: k for k, v in label_map.items()}
    return label_map, inv_label_map


def mediapipe_detection(image, model):
    """Run MediaPipe Holistic."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_keypoints(results) -> np.ndarray:
    """
    Trích xuất keypoints Y HỆT lúc train.
    Output shape: (225,)
    """
    # 1. Pose (33 điểm)
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(N_POSE_LANDMARKS * 3)

    # 2. Left Hand (21 điểm)
    if results.left_hand_landmarks:
        left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        left_hand = np.zeros(N_HAND_LANDMARKS * 3)

    # 3. Right Hand (21 điểm)
    if results.right_hand_landmarks:
        right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        right_hand = np.zeros(N_HAND_LANDMARKS * 3)

    return np.concatenate([pose, left_hand, right_hand])


def process_sequence_padding(
    keypoints_sequence: Sequence[np.ndarray], target_len: int = TARGET_SEQUENCE_LENGTH
) -> np.ndarray:
    """
    Xử lý độ dài chuỗi bằng cách PADDING (chèn số 0) hoặc CẮT BỚT.
    Thay thế cho logic Interpolation cũ để khớp với lúc train.
    """
    if not keypoints_sequence:
        return np.zeros((target_len, N_TOTAL_LANDMARKS))

    # Chuyển list thành numpy array
    sequence_array = np.array(keypoints_sequence)
    
    current_len = len(sequence_array)

    if current_len == target_len:
        return sequence_array
    
    if current_len > target_len:
        # Nếu dài quá -> Cắt bớt (Lấy phần cuối thường quan trọng hơn, hoặc lấy đầu tùy data)
        # Ở đây ta lấy 60 frame cuối cùng
        return sequence_array[-target_len:, :]
    else:
        # Nếu ngắn quá -> Chèn thêm số 0 vào cuối (Zero Padding)
        padding_len = target_len - current_len
        padding = np.zeros((padding_len, sequence_array.shape[1]))
        return np.concatenate([sequence_array, padding], axis=0)


def sequence_frames(video_path: str | Path, holistic) -> list[np.ndarray]:
    """Đọc video và trả về list các keypoints thô."""
    sequence_frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Logic skip frame giống lúc prepare data
    # Nếu video dài > 60 frame thì nhảy cóc để lấy mẫu bao quát
    skip = max(1, total_frames // TARGET_SEQUENCE_LENGTH)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Chỉ lấy frame theo bước nhảy
        if frame_count % skip == 0:
            try:
                image, results = mediapipe_detection(frame, holistic)
                # Chỉ lấy khi có pose (người) để lọc nhiễu
                if results.pose_landmarks:
                    keypoints = extract_keypoints(results)
                    sequence_frames.append(keypoints)
                
                # Nếu đã đủ 60 frame thực tế thì dừng (tối ưu tốc độ)
                if len(sequence_frames) == TARGET_SEQUENCE_LENGTH:
                    break
            except Exception:
                continue
        
        frame_count += 1

    cap.release()
    return sequence_frames


@dataclass
class RecognitionResult:
    label: str
    confidence: float
    probabilities: List[float]


class SignLanguageRecognizer:
    def __init__(
        self,
        model_path: str | Path = "Models/best_model_subset.keras", # Đã sửa đường dẫn mặc định chuẩn
        label_map_path: str | Path = "label_map_subset.json",      # Đã sửa đường dẫn mặc định chuẩn
        target_sequence_length: int = TARGET_SEQUENCE_LENGTH,
    ) -> None:
        self.model_path = Path(model_path)
        self.label_map_path = Path(label_map_path)
        
        # Load Model
        print(f"Loading model from {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)
        
        # Load Labels
        _, self.inv_label_map = load_label_map(self.label_map_path)
        self.target_sequence_length = target_sequence_length

    def predict_from_sequence(self, sequence: Sequence[np.ndarray]) -> RecognitionResult:
        if not sequence:
            raise ValueError("Sequence is empty")

        # BƯỚC QUAN TRỌNG: Padding chuẩn
        kp = process_sequence_padding(sequence, target_len=self.target_sequence_length)
        
        # Dự đoán
        predictions = self.model.predict(np.expand_dims(kp, axis=0), verbose=0)[0]
        pred_idx = int(np.argmax(predictions))
        
        pred_label = self.inv_label_map.get(pred_idx, "Unknown")
        confidence = float(predictions[pred_idx])
        
        return RecognitionResult(
            label=pred_label, confidence=confidence, probabilities=predictions.tolist()
        )

    def predict_from_video(
        self,
        video_path: str | Path,
        holistic: Optional[mp_holistic.Holistic] = None,
    ) -> RecognitionResult:
        own_holistic = holistic is None
        holistic_model = holistic or create_holistic()
        try:
            sequence = sequence_frames(video_path, holistic_model)
        finally:
            if own_holistic:
                holistic_model.close()
        return self.predict_from_sequence(sequence)