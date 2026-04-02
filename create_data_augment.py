import os
import json
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm
import random
import re

# Import các hàm chuẩn hóa từ utils.py
from utils import sequence_frames, interpolate_keypoints
from augment_function import (
    inter_hand_distance, scale_keypoints_sequence, rotate_keypoints_sequence,
    translate_keypoints_sequence, time_stretch_keypoints_sequence, solve_2_link_ik_2d_v2
)

mp_holistic = mp.solutions.holistic

def generate_augmented_samples(
    original_sequence,
    augmentation_functions,
    num_samples_to_generate: int,
    max_augs_per_sample: int = 3, 
):
    generated_samples = []
    if not original_sequence or not augmentation_functions:
        return generated_samples

    num_available_augs = len(augmentation_functions)

    for i in range(num_samples_to_generate):
        current_sequence = [kp.copy() if isinstance(kp, np.ndarray) else kp for kp in original_sequence] 

        num_augs_to_apply = random.randint(1, min(max_augs_per_sample, num_available_augs))
        selected_aug_funcs_indices = random.sample(range(num_available_augs), num_augs_to_apply)
        selected_aug_funcs = [augmentation_functions[idx] for idx in selected_aug_funcs_indices]
        random.shuffle(selected_aug_funcs)

        for aug_func in selected_aug_funcs:
            current_sequence = aug_func(current_sequence)
            if not current_sequence or all(frame is None for frame in current_sequence):
                break

        if not current_sequence or all(frame is None for frame in current_sequence):
            continue 

        generated_samples.append(current_sequence)

    return generated_samples

def sanitize_foldername(name):
    """Loại bỏ các ký tự không hợp lệ cho tên file/folder trên Windows"""
    safe_name = re.sub(r'[<>:"/\\|?*]', '', name)
    return safe_name.strip('. ')

DATASET_PATH = 'Dataset'
LOG_PATH = 'Logs'
BASE_DATA_PATH = 'Data'

os.makedirs(LOG_PATH, exist_ok=True)
label_file = os.path.join(DATASET_PATH, 'Text', 'label.csv')
video_folder = os.path.join(DATASET_PATH, 'Videos')
df = pd.read_csv(label_file)

selected_actions = sorted(df['LABEL'].unique())
label_map = {action: idx for idx, action in enumerate(selected_actions)}

with open(os.path.join(LOG_PATH, 'label_map.json'), 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)

# 1. KHÔNG CHIA SPLIT - DỒN 100% VÀO TRAIN
train_df = df.copy()

splits = {
    'Train': train_df
}

print(f"Tổng số sub-class: {len(label_map)} | Đưa 100% vào Train để học các biến thể.")

augmentations = [
    scale_keypoints_sequence, rotate_keypoints_sequence,
    translate_keypoints_sequence, time_stretch_keypoints_sequence, inter_hand_distance
]

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for split_name, split_data in splits.items():
        print(f"\n--- Đang xử lý tập {split_name} ---")
        
        for _, row in tqdm(split_data.iterrows(), total=len(split_data)):
            action = row['LABEL']
            video_file = row['VIDEO']
            label = label_map[action]
            
            safe_action = sanitize_foldername(action)
            action_path = os.path.join(BASE_DATA_PATH, split_name, safe_action)
            os.makedirs(action_path, exist_ok=True)
            
            idx = len([f for f in os.listdir(action_path) if f.endswith('.npz')])
            video_path = os.path.join(video_folder, video_file)

            if not os.path.exists(video_path):
                continue
            
            frame_lists = sequence_frames(video_path, holistic)
            if not frame_lists:
                continue

            sequences_to_save = []
            
            if split_name == 'Train':
                augmenteds = generate_augmented_samples(frame_lists, augmentations, 50, 2)
                sequences_to_save.extend(augmenteds)
            
            sequences_to_save.append(frame_lists)

            for seq_raw in sequences_to_save:
                seq_interpolated = interpolate_keypoints(seq_raw)
                
                if seq_interpolated is None or np.isnan(seq_interpolated).any():
                    continue

                file_path = os.path.join(action_path, f'{idx}.npz')
                np.savez(file_path, sequence=seq_interpolated.astype(np.float32), label=label)
                idx += 1

print("\nHOÀN THÀNH TẠO DỮ LIỆU!")