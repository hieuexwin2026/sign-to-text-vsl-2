import numpy as np
import pandas as pd
import os
import glob

def export_npz_to_csv(folder_path, output_csv_path):
    print(f"Đang đọc dữ liệu từ: {folder_path}...")
    
    # Tìm tất cả các file .npz trong thư mục
    npz_files = glob.glob(os.path.join(folder_path, '*.npz'))
    
    if not npz_files:
        print("Không tìm thấy file .npz nào trong thư mục!")
        return

    # Sắp xếp file theo thứ tự tên số (0.npz, 1.npz, 2.npz...)
    npz_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    all_dataframes = []

    for file_path in npz_files:
        file_name = os.path.basename(file_path)
        data = np.load(file_path)
        sequence = data['sequence'] # Shape: (60, 201)
        
        # Tạo bảng (DataFrame) cho file hiện tại
        df = pd.DataFrame(sequence)
        
        # Đặt tên cho 201 cột tọa độ để dễ nhìn trong Excel
        # Index 0-74: Pose, 75-137: Tay trái, 138-200: Tay phải
        col_names = []
        for i in range(sequence.shape[1]):
            if i < 75:
                col_names.append(f"Pose_{i}")
            elif i < 138:
                col_names.append(f"LeftHand_{i}")
            else:
                col_names.append(f"RightHand_{i}")
                
        df.columns = col_names
        
        # Thêm 2 cột thông tin ở đầu bảng: Tên file và Thứ tự Frame
        df.insert(0, 'Frame_Index', range(sequence.shape[0]))
        df.insert(0, 'File_Name', file_name)
        
        all_dataframes.append(df)

    # Nối tất cả các bảng nhỏ thành một bảng lớn
    final_df = pd.concat(all_dataframes, ignore_index=True)

    # Xuất ra file CSV (Mở bằng Excel bình thường)
    final_df.to_csv(output_csv_path, index=False)
    print(f"✅ Đã gộp {len(npz_files)} file .npz ({final_df.shape[0]} frames).")
    print(f"✅ Đã xuất thành công ra file: {output_csv_path}")

# --- PHẦN CHẠY THỬ ---
# Điền đường dẫn thư mục chứa từ bạn muốn xem
thu_muc_can_xem = r"Data\Train\Ma Cao_1"

# Tên file Excel (csv) xuất ra
ten_file_xuat = "du_lieu_phuong_tay.csv"

export_npz_to_csv(thu_muc_can_xem, ten_file_xuat)