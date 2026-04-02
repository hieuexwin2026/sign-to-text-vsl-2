import tensorflow as tf

# Thay đổi đường dẫn đến file .keras của bạn
model_path = 'Models/checkpoints/final_model.keras'

try:
    # Tải mô hình
    model = tf.keras.models.load_model(model_path)
    
    # In ra cấu trúc của mô hình
    model.summary()
    
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")