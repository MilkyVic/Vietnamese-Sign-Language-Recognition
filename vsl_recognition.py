"""Reusable Vietnamese Sign Language recognition helpers.

This module isolates the heavy-weight TensorFlow model loading and
MediaPipe-based preprocessing so different front-ends (Streamlit, React,
FastAPI, etc.) can share one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import mediapipe as mp
import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf

mp_holistic = mp.solutions.holistic

N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS = 21
N_TOTAL_LANDMARKS = N_UPPER_BODY_POSE_LANDMARKS + 2 * N_HAND_LANDMARKS
TARGET_SEQUENCE_LENGTH = 60


def create_holistic(
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    """Create a MediaPipe Holistic model with sane defaults."""

    return mp_holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def load_label_map(label_map_path: Path) -> tuple[dict, dict]:
    path = Path(label_map_path)
    with path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}
    return label_map, inv_label_map


def mediapipe_detection(image, model):
    """Run MediaPipe Holistic on a BGR image and return results."""

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_keypoints(results) -> np.ndarray:
    pose_kps = np.zeros((N_UPPER_BODY_POSE_LANDMARKS, 3))
    left_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))
    right_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))

    if results and results.pose_landmarks:
        for i in range(N_UPPER_BODY_POSE_LANDMARKS):
            if i < len(results.pose_landmarks.landmark):
                res = results.pose_landmarks.landmark[i]
                pose_kps[i] = [res.x, res.y, res.z]

    if results and results.left_hand_landmarks:
        left_hand_kps = np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
        )

    if results and results.right_hand_landmarks:
        right_hand_kps = np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        )

    keypoints = np.concatenate([pose_kps, left_hand_kps, right_hand_kps])
    return keypoints.flatten()


def interpolate_keypoints(
    keypoints_sequence: Sequence[np.ndarray], target_len: int = TARGET_SEQUENCE_LENGTH
) -> np.ndarray:
    """Interpolate an arbitrary-length sequence to a fixed number of frames."""

    if not keypoints_sequence:
        raise ValueError("Keypoints sequence is empty")

    original_times = np.linspace(0, 1, len(keypoints_sequence))
    target_times = np.linspace(0, 1, target_len)

    num_features = keypoints_sequence[0].shape[0]
    interpolated_sequence = np.zeros((target_len, num_features))

    for feature_idx in range(num_features):
        feature_values = [frame[feature_idx] for frame in keypoints_sequence]

        interpolator = interp1d(
            original_times,
            feature_values,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        interpolated_sequence[:, feature_idx] = interpolator(target_times)

    return interpolated_sequence


def sequence_frames(video_path: str | Path, holistic) -> list[np.ndarray]:
    """Extract sparse frames from a video and convert them to keypoints."""

    sequence_frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // 100)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % step != 0:
            continue

        try:
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            if keypoints is not None:
                sequence_frames.append(keypoints)
        except Exception:
            continue

    cap.release()
    return sequence_frames


@dataclass
class RecognitionResult:
    label: str
    confidence: float
    probabilities: List[float]


class SignLanguageRecognizer:
    """Wrapper that keeps the TF model in memory for quick predictions."""

    def __init__(
        self,
        model_path: str | Path = "Models/checkpoints_gpu/best_model.keras",
        label_map_path: str | Path = "Logs/label_map.json",
        target_sequence_length: int = TARGET_SEQUENCE_LENGTH,
    ) -> None:
        self.model_path = Path(model_path)
        self.label_map_path = Path(label_map_path)
        self.model = tf.keras.models.load_model(self.model_path)
        _, self.inv_label_map = load_label_map(self.label_map_path)
        self.target_sequence_length = target_sequence_length

    def predict_from_sequence(self, sequence: Sequence[np.ndarray]) -> RecognitionResult:
        if not sequence:
            raise ValueError("Sequence is empty")

        kp = interpolate_keypoints(sequence, target_len=self.target_sequence_length)
        predictions = self.model.predict(np.expand_dims(kp, axis=0), verbose=0)[0]
        pred_idx = int(np.argmax(predictions))
        pred_label = self.inv_label_map[pred_idx]
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


__all__ = [
    "RecognitionResult",
    "SignLanguageRecognizer",
    "create_holistic",
    "extract_keypoints",
    "interpolate_keypoints",
    "mediapipe_detection",
    "sequence_frames",
]

