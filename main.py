import streamlit as st
import numpy as np
import tensorflow as tf
import tempfile
import os
import cv2
import mediapipe as mp
import time

# IMPORT TỪ FILE UTILS
from utils import sequence_frames, interpolate_keypoints, mediapipe_detection, extract_keypoints

st.set_page_config(page_title="VSL Prediction", layout="centered")
st.title("DỰ ĐOÁN NGÔN NGỮ KÝ HIỆU")

mp_holistic = mp.solutions.holistic

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('Models/checkpoints/final_model.keras')

@st.cache_data
def load_label_map():
    import json
    with open('Logs/label_map.json', 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}
    return label_map, inv_label_map

model = load_model()
label_map, inv_label_map = load_label_map()

def process_webcam_to_sequence():
    cap = cv2.VideoCapture(0)
    st.write("⏳ Đang chuẩn bị... Bắt đầu trong 1.5 giây...")
    time.sleep(1.5)
    
    st.write("🎥 Đang ghi hình trong 4 giây...")
    sequence = []
    start_time = time.time()

    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Không thể truy cập webcam")
            break
        elapsed_time = time.time() - start_time
        if elapsed_time > 4:
            break
            
        # Dùng hàm đồng bộ từ utils
        image, results = mediapipe_detection(frame, holistic)
        keypoints = extract_keypoints(results)
        
        if keypoints is not None:
            sequence.append(keypoints)

        stframe.image(image, channels="BGR", caption="Webcam feed", use_container_width=True)

    cap.release()
    return sequence

# Giao diện người dùng
input_mode = st.radio("Chọn nguồn đầu vào:", ["🎞️ Video file", "📷 Webcam"])
sequence = None

if input_mode == "🎞️ Video file":
    uploaded_file = st.file_uploader("Tải lên video (.mp4, .avi)", type=["mp4", "avi"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        st.video(tmp_path)
        if st.button("🔍 Dự đoán từ video"):
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                # Dùng hàm đồng bộ từ utils
                sequence = sequence_frames(tmp_path, holistic)

elif input_mode == "📷 Webcam":
    st.warning("Nhấn nút bên dưới để bắt đầu ghi hình từ webcam.")
    if st.button("📸 Ghi và dự đoán"):
        sequence = process_webcam_to_sequence()

# Dự đoán
if sequence is not None:
    kp = interpolate_keypoints(sequence)

    preds = model.predict(np.expand_dims(kp, axis=0))[0]

    top_k = 10
    top_indices = np.argsort(preds)[::-1][:top_k]

    st.success(f"✅ Nhãn dự đoán chính: **{inv_label_map[top_indices[0]]}**")

    st.markdown("### 🔝 Top 10 nhãn tương đồng nhất:")
    for rank, idx in enumerate(top_indices, start=1):
        label = inv_label_map[idx]
        prob = preds[idx] * 100
        st.write(f"{rank}. **{label}** — {prob:.2f}%")