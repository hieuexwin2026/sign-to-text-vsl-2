import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
import pandas as pd
import os
import re
from tqdm import tqdm

# ================= IMPORT TỪ UTILS =================
# Xóa bỏ các hàm tự viết lặp lại, gọi thẳng từ utils.py để đồng bộ 100%
from utils import sequence_frames, interpolate_keypoints

# ================= CONFIG =================
MODEL_PATH = "Models/checkpoints/final_model.keras"
LABEL_MAP_PATH = "Logs/label_map.json"
LABEL_CSV_PATH = "Dataset/Text/label.csv"
VIDEO_FOLDER = "Dataset/Videos"

SEQUENCE_LEN = 60
TOP_K = 5
LOG_DIR = "Logs"
LOG_FILE = os.path.join(LOG_DIR, "wrong_predictions.txt")
# =========================================

os.makedirs(LOG_DIR, exist_ok=True)
mp_holistic = mp.solutions.holistic

# ================= LOAD =================
print("🔹 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)

inv_label_map = {v: k for k, v in label_map.items()}

df = pd.read_csv(LABEL_CSV_PATH)
video_to_label = dict(zip(df["VIDEO"], df["LABEL"]))


# ================= TEST LOOP =================
videos = [
    f for f in os.listdir(VIDEO_FOLDER)
    if f.lower().endswith((".mp4", ".avi", ".mov"))
]

total = 0
correct = 0
wrong_logs = []

print(f"\n🎞️ Found {len(videos)} videos\n")

# Mở một phiên Holistic duy nhất để dùng cho tất cả video (tăng tốc độ chạy)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for video in tqdm(videos, desc="Testing videos"):
        if video not in video_to_label:
            continue

        gt_label_raw = video_to_label[video]
        video_path = os.path.join(VIDEO_FOLDER, video)

        # Sử dụng hàm chuẩn từ utils.py
        raw_seq = sequence_frames(video_path, holistic)
        seq = interpolate_keypoints(raw_seq)

        if seq is None or np.isnan(seq).any():
            continue

        preds = model.predict(seq[None, ...], verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        pred_label_raw = inv_label_map[pred_idx]

        # ================= LOGIC CHẤM ĐIỂM THÔNG MINH =================
        # Gọt bỏ phần hậu tố _1, _2... để lấy nghĩa gốc của từ
        gt_label_clean = re.sub(r'_\d+$', '', gt_label_raw)
        pred_label_clean = re.sub(r'_\d+$', '', pred_label_raw)

        total += 1

        # So sánh dựa trên nghĩa gốc
        if pred_label_clean == gt_label_clean:
            correct += 1
        else:
            topk = np.argsort(preds)[::-1][:TOP_K]
            wrong_logs.append(
                f"VIDEO: {video}\n"
                f"  TRUE : {gt_label_raw} -> Ý nghĩa: {gt_label_clean}\n"
                f"  PRED : {pred_label_raw} -> Ý nghĩa: {pred_label_clean}\n"
                f"  CONF : {preds[pred_idx]:.4f}\n"
                f"  TOP{TOP_K}: {', '.join([inv_label_map[i] for i in topk])}\n"
                f"{'-'*40}\n"
            )

# ================= RESULT =================
accuracy = correct / total * 100 if total > 0 else 0

print("\n" + "=" * 60)
print(f"✅ TOTAL TESTED : {total}")
print(f"🎯 CORRECT      : {correct}")
print(f"❌ WRONG        : {total - correct}")
print(f"📊 ACCURACY     : {accuracy:.2f}%")
print("=" * 60)

# ================= SAVE TXT LOG =================
if wrong_logs:
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.writelines(wrong_logs)
    print(f"\n📄 Wrong predictions saved to: {LOG_FILE}")
else:
    print("\n🎉 No wrong predictions!")

# import cv2
# import numpy as np
# import tensorflow as tf
# import mediapipe as mp
# import json
# import pandas as pd
# import os
# from tqdm import tqdm

# # ================= IMPORT TỪ UTILS =================
# # Xóa bỏ các hàm tự viết lặp lại, gọi thẳng từ utils.py để đồng bộ 100%
# from utils import sequence_frames, interpolate_keypoints

# # ================= CONFIG =================
# MODEL_PATH = "Models/checkpoints/final_model.keras"
# LABEL_MAP_PATH = "Logs/label_map.json"
# LABEL_CSV_PATH = "Dataset/Text/label.csv"
# VIDEO_FOLDER = "Dataset/Videos"

# SEQUENCE_LEN = 60
# TOP_K = 5
# LOG_DIR = "Logs"
# LOG_FILE = os.path.join(LOG_DIR, "wrong_predictions.txt")
# # =========================================

# os.makedirs(LOG_DIR, exist_ok=True)
# mp_holistic = mp.solutions.holistic

# # ================= LOAD =================
# print("🔹 Loading model...")
# model = tf.keras.models.load_model(MODEL_PATH)

# with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
#     label_map = json.load(f)

# inv_label_map = {v: k for k, v in label_map.items()}

# df = pd.read_csv(LABEL_CSV_PATH)
# video_to_label = dict(zip(df["VIDEO"], df["LABEL"]))


# # ================= TEST LOOP =================
# videos = [
#     f for f in os.listdir(VIDEO_FOLDER)
#     if f.lower().endswith((".mp4", ".avi", ".mov"))
# ]

# total = 0
# correct = 0
# wrong_logs = []

# print(f"\n🎞️ Found {len(videos)} videos\n")

# # Mở một phiên Holistic duy nhất để tăng tốc độ
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     for video in tqdm(videos, desc="Testing videos"):
#         if video not in video_to_label:
#             continue

#         gt_label = video_to_label[video]
#         video_path = os.path.join(VIDEO_FOLDER, video)

#         # Sử dụng hàm chuẩn từ utils.py
#         raw_seq = sequence_frames(video_path, holistic)
#         seq = interpolate_keypoints(raw_seq)

#         if seq is None or np.isnan(seq).any():
#             continue

#         preds = model.predict(seq[None, ...], verbose=0)[0]
#         pred_idx = int(np.argmax(preds))
#         pred_label = inv_label_map[pred_idx]

#         total += 1

#         # ================= SO SÁNH CHÍNH XÁC (KHÔNG GỌT ĐUÔI) =================
#         if pred_label == gt_label:
#             correct += 1
#         else:
#             topk = np.argsort(preds)[::-1][:TOP_K]
#             wrong_logs.append(
#                 f"VIDEO: {video}\n"
#                 f"  TRUE : {gt_label}\n"
#                 f"  PRED : {pred_label}\n"
#                 f"  CONF : {preds[pred_idx]:.4f}\n"
#                 f"  TOP{TOP_K}: {', '.join([inv_label_map[i] for i in topk])}\n"
#                 f"{'-'*40}\n"
#             )

# # ================= RESULT =================
# accuracy = correct / total * 100 if total > 0 else 0

# print("\n" + "=" * 60)
# print(f"✅ TOTAL TESTED : {total}")
# print(f"🎯 CORRECT      : {correct}")
# print(f"❌ WRONG        : {total - correct}")
# print(f"📊 ACCURACY     : {accuracy:.2f}%")
# print("=" * 60)

# # ================= SAVE TXT LOG =================
# if wrong_logs:
#     with open(LOG_FILE, "w", encoding="utf-8") as f:
#         f.writelines(wrong_logs)
#     print(f"\n📄 Wrong predictions saved to: {LOG_FILE}")
# else:
#     print("\n🎉 No wrong predictions!")