import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from scipy.interpolate import interp1d

mp_holistic = mp.solutions.holistic
N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS = 21

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
    """Đọc TẤT CẢ các frame có thể nhận diện được, KHÔNG nhảy bước"""
    sequence = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        try:
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            if keypoints is not None:
                sequence.append(keypoints)
        except Exception:
            continue
    cap.release()
    return sequence

def interpolate_keypoints(keypoints_sequence, target_len=60):
    """Nội suy tuyến tính (Linear) + Fill N/A để tránh văng đồ thị khi mất tọa độ"""
    if not keypoints_sequence or len(keypoints_sequence) == 0:
        return None

    seq = np.array(keypoints_sequence)
    
    # Làm sạch: Lấp đầy các giá trị 0 (không nhận diện được) bằng frame ngay trước/sau nó
    df = pd.DataFrame(seq).replace(0.0, np.nan).ffill().bfill().fillna(0.0)
    seq = df.values

    original_times = np.linspace(0, 1, len(seq))
    target_times = np.linspace(0, 1, target_len)
    num_features = seq.shape[1]
    interpolated_sequence = np.zeros((target_len, num_features))

    for feature_idx in range(num_features):
        interpolator = interp1d(
            original_times, seq[:, feature_idx],
            kind='linear', # Đổi sang linear an toàn hơn
            bounds_error=False,
            fill_value="extrapolate"
        )
        interpolated_sequence[:, feature_idx] = interpolator(target_times)

    return interpolated_sequence