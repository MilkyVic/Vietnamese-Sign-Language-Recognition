import numpy as np
import json
import os

# --- CẤU HÌNH ---
INPUT_DIR = "NPY"  # Folder chứa dữ liệu npy của bạn
OUTPUT_DIR = "Kinesis3/public/assets/skeletons" # Folder public của Frontend React
# Danh sách 4 lớp bạn muốn làm demo
TARGET_CLASSES = ["tôi yêu bạn", "yêu nước việt nam", "ngày Người khuyết tật Việt Nam 18-4", "bạn ấy yêu tôi"] 

# Tạo folder nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)

def npy_to_json(class_name):
    # Đường dẫn tới folder của lớp đó
    class_path = os.path.join(INPUT_DIR, class_name)
    
    # Kiểm tra folder tồn tại không
    if not os.path.exists(class_path):
        print(f"⚠️ Không tìm thấy folder: {class_name}")
        return

    # Lấy danh sách file .npy
    files = [f for f in os.listdir(class_path) if f.endswith('.npy')]
    if not files:
        print(f"⚠️ Không có file .npy trong {class_name}")
        return
    
    # MẸO: Lấy file đầu tiên hoặc file ngẫu nhiên làm mẫu
    # Bạn có thể mở folder ra xem file nào dung lượng chuẩn nhất thì điền tên vào đây
    sample_file = files[0] 
    file_path = os.path.join(class_path, sample_file)
    
    print(f"🔄 Đang xử lý: {class_name} (File mẫu: {sample_file})")
    
    # Load dữ liệu (Shape: [60, 225])
    data = np.load(file_path)
    
    frames_export = []
    
    # Duyệt qua từng frame (tối đa 60 frame)
    for frame_idx in range(data.shape[0]):
        row = data[frame_idx]
        
        # Cắt chuỗi 225 điểm thành các phần:
        # Pose (33 điểm * 3), Tay Trái (21 * 3), Tay Phải (21 * 3)
        pose_flat = row[:99]
        lh_flat = row[99:162]
        rh_flat = row[162:]
        
        # Hàm helper chuyển mảng phẳng thành list object {x, y}
        def to_points(flat_arr):
            points = []
            for i in range(0, len(flat_arr), 3):
                # Lưu ý: Chỉ lấy x, y. Bỏ z.
                x = float(flat_arr[i])
                y = float(flat_arr[i+1])
                points.append({"x": x, "y": y})
            return points

        frames_export.append({
            "pose": to_points(pose_flat),
            "left_hand": to_points(lh_flat),
            "right_hand": to_points(rh_flat)
        })

    # Tên file json output (viết liền không dấu cho dễ gọi)
    # Ví dụ: "tôi yêu bạn" -> "toi_yeu_ban.json"
    # Bạn có thể đặt tên thủ công hoặc map
    safe_name = class_name.replace(" ", "_") # Chuyển khoảng trắng thành _
    save_path = os.path.join(OUTPUT_DIR, f"{safe_name}.json")
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(frames_export, f)
    
    print(f"✅ Đã xuất: {save_path}")

# Chạy vòng lặp
for cls in TARGET_CLASSES:
    npy_to_json(cls)