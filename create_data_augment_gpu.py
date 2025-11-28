import os
import re
import json
import random
import math
from datetime import datetime

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


mp_holistic = mp.solutions.holistic
N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS = 21
N_TOTAL_LANDMARKS = N_UPPER_BODY_POSE_LANDMARKS + N_HAND_LANDMARKS + N_HAND_LANDMARKS
LEFT_HAND_START = N_UPPER_BODY_POSE_LANDMARKS
RIGHT_HAND_START = LEFT_HAND_START + N_HAND_LANDMARKS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("No CUDA device detected. Please run on a machine with GPU support.")


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_keypoints(results):
    pose_kps = np.zeros((N_UPPER_BODY_POSE_LANDMARKS, 3))
    left_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))
    right_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))

    if results and results.pose_landmarks:
        for i in range(N_UPPER_BODY_POSE_LANDMARKS):
            if i < len(results.pose_landmarks.landmark):
                res = results.pose_landmarks.landmark[i]
                pose_kps[i] = [res.x, res.y, res.z]

    if results and results.left_hand_landmarks:
        left_hand_kps = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])

    if results and results.right_hand_landmarks:
        right_hand_kps = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])

    keypoints = np.concatenate([pose_kps, left_hand_kps, right_hand_kps])
    return keypoints.flatten()


def sequence_frames(video_path, holistic):
    frames = []
    cap = cv2.VideoCapture(video_path)
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
                frames.append(keypoints)
        except Exception:
            continue

    cap.release()
    return frames


def sanitize_action_name(action_name: str) -> str:
    sanitized = action_name.strip()
    sanitized = re.sub(r'[\\/:*?"<>|\r\n]+', '_', sanitized)
    sanitized = sanitized.rstrip('. ')
    return sanitized or "action"


def create_action_folder(base_path, action, action_dirs):
    if action not in action_dirs:
        base = sanitize_action_name(action)
        candidate = base
        suffix = 1
        while candidate in action_dirs.values():
            candidate = f"{base}_{suffix}"
            suffix += 1
        action_dirs[action] = candidate

    folder_name = action_dirs[action]
    action_path = os.path.join(base_path, folder_name)
    existed = os.path.isdir(action_path)
    os.makedirs(action_path, exist_ok=True)
    return action_path, existed


def list_to_tensor(sequence):
    frames = []
    for frame in sequence:
        if frame is None:
            frames.append(torch.zeros(N_TOTAL_LANDMARKS * 3, device=device))
            continue

        frame_arr = np.asarray(frame)
        if frame_arr.size != N_TOTAL_LANDMARKS * 3:
            continue
        frames.append(torch.from_numpy(frame_arr).to(device=device, dtype=torch.float32))

    if not frames:
        return None

    stacked = torch.stack(frames)
    return stacked.view(-1, N_TOTAL_LANDMARKS, 3)


def tensor_to_numpy(sequence_tensor):
    return sequence_tensor.detach().cpu().view(sequence_tensor.shape[0], -1).numpy()


def _valid_mask(sequence_tensor):
    return (sequence_tensor.abs().sum(dim=2) > 0)


def scale_sequence_gpu(sequence_tensor, scale_range=(0.7, 1.26)):
    tensor = sequence_tensor.clone()
    xy = tensor[..., :2]
    mask = _valid_mask(tensor)
    mask_unsq = mask.unsqueeze(-1)

    counts = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
    centers = (xy * mask_unsq).sum(dim=1) / counts

    scale = torch.empty(1, device=device).uniform_(*scale_range)
    xy = torch.where(mask_unsq, (xy - centers.unsqueeze(1)) * scale + centers.unsqueeze(1), xy)
    tensor[..., :2] = xy
    return tensor


def rotate_sequence_gpu(sequence_tensor, angle_range=(-15.0, 15.0)):
    tensor = sequence_tensor.clone()
    xy = tensor[..., :2]
    mask = _valid_mask(tensor)
    mask_unsq = mask.unsqueeze(-1)

    counts = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
    centers = (xy * mask_unsq).sum(dim=1) / counts

    angle = math.radians(random.uniform(angle_range[0], angle_range[1]))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rot = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=device, dtype=torch.float32)

    shifted = xy - centers.unsqueeze(1)
    rotated = shifted @ rot.T
    xy = torch.where(mask_unsq, rotated + centers.unsqueeze(1), xy)
    tensor[..., :2] = xy
    return tensor


def translate_sequence_gpu(sequence_tensor, translate_range=(-0.05, 0.05)):
    tensor = sequence_tensor.clone()
    xy = tensor[..., :2]
    mask = _valid_mask(tensor).unsqueeze(-1)
    shift = torch.empty(2, device=device).uniform_(translate_range[0], translate_range[1])
    xy = torch.where(mask, xy + shift, xy)
    xy = torch.clamp(xy, 0.0, 1.0)
    tensor[..., :2] = xy
    return tensor


def adjust_inter_hand_distance_gpu(sequence_tensor, distance_range=(0.9, 1.25)):
    tensor = sequence_tensor.clone()
    left = tensor[:, LEFT_HAND_START:LEFT_HAND_START + N_HAND_LANDMARKS, :2]
    right = tensor[:, RIGHT_HAND_START:RIGHT_HAND_START + N_HAND_LANDMARKS, :2]
    left_mask = (left.abs().sum(dim=2) > 0)
    right_mask = (right.abs().sum(dim=2) > 0)

    left_counts = left_mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)
    right_counts = right_mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)
    left_center = (left * left_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / left_counts
    right_center = (right * right_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / right_counts

    left_center = left_center.squeeze(1)
    right_center = right_center.squeeze(1)

    frame_has_both = (left_mask.sum(dim=1) > 0) & (right_mask.sum(dim=1) > 0)
    if not torch.any(frame_has_both):
        return tensor

    factor = random.uniform(distance_range[0], distance_range[1])
    delta = torch.zeros_like(left_center)
    delta[frame_has_both] = (left_center - right_center)[frame_has_both] * (factor - 1.0) * 0.5
    delta_expanded = delta.unsqueeze(1).expand(-1, N_HAND_LANDMARKS, -1)
    left_mask_expand = left_mask.unsqueeze(-1).expand(-1, -1, 2)
    right_mask_expand = right_mask.unsqueeze(-1).expand(-1, -1, 2)

    left = torch.where(left_mask_expand, left + delta_expanded, left)
    right = torch.where(right_mask_expand, right - delta_expanded, right)
    tensor[:, LEFT_HAND_START:LEFT_HAND_START + N_HAND_LANDMARKS, :2] = left
    tensor[:, RIGHT_HAND_START:RIGHT_HAND_START + N_HAND_LANDMARKS, :2] = right
    return tensor


def time_stretch_sequence_gpu(sequence_tensor, speed_range=(0.8, 1.2)):
    factor = random.uniform(speed_range[0], speed_range[1])
    if factor <= 0:
        return sequence_tensor.clone()

    current_len = sequence_tensor.shape[0]
    target_len = max(2, int(round(current_len / factor)))
    flat = sequence_tensor.view(current_len, -1).transpose(0, 1).unsqueeze(0)
    stretched = F.interpolate(flat, size=target_len, mode='linear', align_corners=True)
    stretched = stretched.squeeze(0).transpose(0, 1).contiguous().view(target_len, N_TOTAL_LANDMARKS, 3)
    return stretched


def interpolate_tensor(sequence_tensor, target_len=60):
    current_len = sequence_tensor.shape[0]
    if current_len == target_len:
        return sequence_tensor
    flat = sequence_tensor.view(current_len, -1).transpose(0, 1).unsqueeze(0)
    interp = F.interpolate(flat, size=target_len, mode='linear', align_corners=True)
    interp = interp.squeeze(0).transpose(0, 1).contiguous().view(target_len, N_TOTAL_LANDMARKS, 3)
    return interp


def generate_augmented_samples_gpu(base_tensor, augmentation_functions, num_samples, max_augs_per_sample=3):
    if base_tensor is None:
        return []

    generated = []
    num_funcs = len(augmentation_functions)
    if num_funcs == 0:
        return [base_tensor.clone()]

    for _ in range(num_samples):
        current = base_tensor.clone()
        num_steps = random.randint(1, min(max_augs_per_sample, num_funcs))
        for func in random.sample(augmentation_functions, num_steps):
            current = func(current)
        generated.append(current)

    return generated


def has_existing_sequences(action_path):
    try:
        for entry in os.scandir(action_path):
            if entry.is_file() and entry.name.endswith(".npz"):
                return True
    except FileNotFoundError:
        return False
    return False


class GetTime:
    def __init__(self):
        self.starttime = datetime.now()

    def get_time(self):
        return datetime.now() - self.starttime


DATA_PATH = os.path.join('Data_gpu')
DATASET_PATH = os.path.join('Dataset')
LOG_PATH = os.path.join('Logs')
SEQUENCE_LENGTH = 60

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

label_file = os.path.join(DATASET_PATH, 'Text', 'label.csv')
video_folder = os.path.join(DATASET_PATH, 'Videos')
df = pd.read_csv(label_file)

selected_actions = sorted(df['LABEL'].unique())
label_map = {action: idx for idx, action in enumerate(selected_actions)}
action_directories = {}

label_map_path = os.path.join(LOG_PATH, 'label_map.json')
with open(label_map_path, 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)

print(f"Selected {len(selected_actions)} actions")

augmentation_functions = [
    scale_sequence_gpu,
    rotate_sequence_gpu,
    translate_sequence_gpu,
    time_stretch_sequence_gpu,
    adjust_inter_hand_distance_gpu,
]

timer = GetTime()
print(f"{datetime.now()} Start processing data with GPU on {device}")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    action_position = {action: idx + 1 for idx, action in enumerate(pd.unique(df['LABEL']))}

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Process actions'):
        action = row['LABEL']
        video_file = row['VIDEO']
        label = label_map[action]

        action_path, existed = create_action_folder(DATA_PATH, action, action_directories)
        if existed and has_existing_sequences(action_path):
            print(f"Skip action '{action}' – augmented data already exists.")
            continue
        video_path = os.path.join(video_folder, video_file)

        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue

        frame_sequence = sequence_frames(video_path, holistic)
        base_tensor = list_to_tensor(frame_sequence)
        if base_tensor is None:
            print(f"No valid frames for action {action}")
            continue

        augmented_tensors = generate_augmented_samples_gpu(base_tensor, augmentation_functions, num_samples=1000, max_augs_per_sample=5)
        augmented_tensors.append(base_tensor)

        idx = 0
        for seq_tensor in augmented_tensors:
            seq_tensor = interpolate_tensor(seq_tensor, SEQUENCE_LENGTH)
            sequence_np = tensor_to_numpy(seq_tensor)
            file_path = os.path.join(action_path, f'{idx}.npz')
            np.savez(file_path, sequence=sequence_np, label=label)
            idx += 1

        print(f"Action {action_position[action]}/{len(selected_actions)} : {action} - Time: {timer.get_time()}")

print("-" * 50)
print("GPU DATA PROCESSING COMPLETED.")
