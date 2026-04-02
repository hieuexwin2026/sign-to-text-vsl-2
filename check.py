import numpy as np
import os
import glob

def analyze_original_npz(folder_path):
    if not os.path.exists(folder_path):
        print(f"Không tìm thấy thư mục: {folder_path}")
        return
    
    # Tìm tất cả các file .npz trong thư mục
    npz_files = glob.glob(os.path.join(folder_path, '*.npz'))
    if not npz_files:
        print("Thư mục rỗng, không có file .npz")
        return
        
    # Sắp xếp để lấy file có index lớn nhất (file gốc chưa augment)
    npz_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    target_file = npz_files[-1]
    
    # Load file npz
    data = np.load(target_file)
    sequence = data['sequence'] 
    
    print(f"--- PHÂN TÍCH FILE GỐC: {target_file} ---")
    print(f"Kích thước mảng (Shape): {sequence.shape}")
    
    total_zeros = np.sum(sequence == 0.0)
    total_elements = sequence.size
    print(f"Tổng số giá trị 0.0: {total_zeros}/{total_elements} ({total_zeros/total_elements*100:.2f}% dữ liệu là số 0)")
    
    print("\nChi tiết các frame bị MediaPipe mất dấu bàn tay:")
    frames_with_missing_hands = 0
    
    for frame_idx, frame in enumerate(sequence):
        # Trích xuất 2 bàn tay (theo cấu trúc 25 Pose + 21 Left + 21 Right = 67 điểm x 3)
        left_hand = frame[75:138]
        right_hand = frame[138:201]
        
        left_missing = np.all(left_hand == 0.0)
        right_missing = np.all(right_hand == 0.0)
        
        if left_missing or right_missing:
            frames_with_missing_hands += 1
            status = []
            if left_missing: status.append("Mất dấu TAY TRÁI")
            if right_missing: status.append("Mất dấu TAY PHẢI")
            
            print(f" - Frame {frame_idx:02d}: {', '.join(status)}")
            
    print("-" * 40)
    if frames_with_missing_hands == 0:
        print("Dữ liệu rất sạch, không có frame nào bị rỗng hoàn toàn tay.")
    else:
        print(f"=> CẢNH BÁO: Có {frames_with_missing_hands}/{len(sequence)} frame bị mất dấu tay.")
        print("Hàm ffill() và bfill() vô hiệu hóa vì không tìm được frame lân cận có dữ liệu!")

# --- TEST 2 TỪ BỊ LỖI ---
# Lưu ý: Chữ phương Tây_1 cần khớp chính xác tên thư mục được tạo ra
folder_phuong_tay = r"Data\Train\địa chỉ_1"
folder_hoi_dau = r"Data\Train\hói đầu_1"

analyze_original_npz(folder_phuong_tay)
print("\n")
analyze_original_npz(folder_hoi_dau)