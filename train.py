import os
import json
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization, Masking
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from collections import defaultdict
import cv2

# --- CẤU HÌNH ---
# Đảm bảo đường dẫn này đúng với folder bạn vừa chạy prepare_data xong
DATA_DIR = 'Prepared_Data_All' 
LABEL_MAP_PATH = 'label_map.json'
MODELS_DIR = 'Models'

# Hyperparameters
EPOCHS = 200            # Train lâu để model học kỹ
BATCH_SIZE = 32         # Batch size tiêu chuẩn
INPUT_SHAPE = (60, 225) # 60 frames, 225 keypoints
LEARNING_RATE = 1e-3    # Tốc độ học khởi điểm

# --- 1. HÀM AUGMENTATION (Biến hình dữ liệu) ---
def augment_data(data):
    """
    Kỹ thuật Data Augmentation mạnh cho bài toán One-Shot.
    Input: (60, 225)
    Output: (60, 225) đã bị biến đổi nhẹ
    """
    # Copy để không ảnh hưởng dữ liệu gốc
    aug_data = data.copy()
    
    # 1. Uniform Scale (Phóng to/Thu nhỏ cơ thể)
    # Giả lập người đứng gần hoặc xa camera
    scale = np.random.uniform(0.85, 1.15)
    aug_data = aug_data * scale

    # 2. Translation (Dịch chuyển khung hình)
    # Giả lập người đứng lệch trái/phải/lên/xuống
    shift_x = np.random.uniform(-0.1, 0.1)
    shift_y = np.random.uniform(-0.1, 0.1)
    # Chỉ dịch chuyển 2 trục đầu (x, y), giữ nguyên z
    # Cấu trúc landmark: x, y, z lặp lại. Ta cộng shift vào toàn bộ ma trận
    # Tuy nhiên cách đơn giản nhất là cộng random noise vào, 
    # nhưng ở đây ta làm đơn giản là shift toàn bộ.
    aug_data[:, 0::3] += shift_x # Cộng vào tọa độ X
    aug_data[:, 1::3] += shift_y # Cộng vào tọa độ Y

    # 3. Time Stretch (Co giãn thời gian)
    # Giả lập làm động tác nhanh hơn hoặc chậm hơn
    if np.random.rand() > 0.5:
        # Resize chiều thời gian (trục 0)
        # Random độ dài mới từ 50 đến 70 frames
        new_len = int(np.random.uniform(50, 70))
        # Dùng cv2 để resize ma trận 2D (coi như 1 bức ảnh 60x225)
        aug_data = cv2.resize(aug_data, (225, new_len))
        
        # Sau khi resize, phải đưa về lại chuẩn 60 frames
        if new_len < 60:
            # Padding số 0 nếu ngắn hơn
            pad = np.zeros((60 - new_len, 225))
            aug_data = np.vstack([aug_data, pad])
        else:
            # Cắt bớt nếu dài hơn
            aug_data = aug_data[:60, :]

    # 4. Gaussian Noise (Nhiễu ngẫu nhiên)
    # Giả lập camera rung hoặc mediapipe bắt điểm không chuẩn
    noise = np.random.normal(0, 0.005, size=aug_data.shape)
    aug_data = aug_data + noise
    
    return aug_data

# --- 2. DATA GENERATOR ---
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size, dim, n_classes, shuffle=True, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.file_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        for i, k in enumerate(indexes):
            # Load file .npy
            data = np.load(self.file_paths[k])
            
            # Áp dụng Augmentation nếu là tập Train
            if self.augment:
                try:
                    data = augment_data(data)
                except Exception:
                    # Nếu augmentation lỗi (hiếm), giữ nguyên gốc
                    pass
            
            # Đảm bảo shape đúng (phòng hờ)
            if data.shape != self.dim:
                # Resize về chuẩn nếu sai lệch
                data = cv2.resize(data, (self.dim[1], self.dim[0]))

            X[i,] = data
            y[i] = self.labels[k]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

# --- 3. MODEL BUILDER (GRU + Gradient Clipping) ---
def build_model(num_classes):
    model = Sequential()
    
    # 1. Masking: Bỏ qua các giá trị 0 (padding)
    model.add(Masking(mask_value=0.0, input_shape=INPUT_SHAPE))
    
    # 2. GRU Layers (Nhẹ hơn và ổn định hơn LSTM cho data ít)
    model.add(GRU(256, return_sequences=True, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4)) # Dropout cao để chống overfit
    
    model.add(GRU(128, return_sequences=False, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    # 3. Dense Layers
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    
    # 4. Output Layer
    model.add(Dense(num_classes, activation='softmax'))

    # OPTIMIZER QUAN TRỌNG:
    # clipnorm=1.0: Cắt bớt gradient nếu nó quá lớn -> CHỐNG LOSS NHẢY LÊN 400
    optimizer = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    
    # LOSS QUAN TRỌNG:
    # label_smoothing=0.1: Làm mềm nhãn, giúp model không quá tự tin -> CHỐNG OVERFIT
    model.compile(
        optimizer=optimizer, 
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), 
        metrics=['accuracy']
    )
    return model

# --- MAIN PROCESS ---
if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Load Label Map
    if not os.path.exists(LABEL_MAP_PATH):
        print(f"Lỗi: Không tìm thấy {LABEL_MAP_PATH}. Hãy chạy prepare_data.py trước!")
        exit()

    with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    print(f"Loaded {len(label_map)} classes.")

    # 2. Quét file và Chia dữ liệu thủ công (Manual Split)
    print("Scanning files...")
    files = glob.glob(os.path.join(DATA_DIR, '**', '*.npy'), recursive=True)
    
    # Nhóm file theo Class
    label_dict = defaultdict(list)
    for f in files:
        # Lấy tên folder cha làm nhãn (Windows/Linux safe)
        label_name = os.path.basename(os.path.dirname(f))
        if label_name in label_map:
            label_dict[label_name].append(f)

    X_train, y_train = [], []
    X_val, y_val = [], []

    print("Splitting data (One-Shot Logic)...")
    for label_name, paths in label_dict.items():
        lbl_idx = label_map[label_name]
        
        # LOGIC QUAN TRỌNG:
        if len(paths) == 1:
            # Nếu chỉ có 1 video: Dùng cho cả Train (để học biến thể) và Val (để test gốc)
            X_train.extend(paths)
            y_train.append(lbl_idx)
            
            X_val.extend(paths)
            y_val.append(lbl_idx)
        else:
            # Nếu có nhiều video: Chia 80/20
            split = int(len(paths) * 0.8)
            if split == 0: split = 1 # Ít nhất 1 mẫu cho train
            
            X_train.extend(paths[:split])
            y_train.extend([lbl_idx]*len(paths[:split]))
            
            # Phần còn lại cho Val
            val_part = paths[split:]
            if len(val_part) == 0: val_part = [paths[-1]] # Fallback nếu chia hết
            
            X_val.extend(val_part)
            y_val.extend([lbl_idx]*len(val_part))

    print(f"--> Train set: {len(X_train)} files")
    print(f"--> Val set:   {len(X_val)} files")

    # 3. Khởi tạo Generators
    # Train: CÓ Augment
    train_gen = DataGenerator(
        X_train, y_train, 
        batch_size=BATCH_SIZE, 
        dim=INPUT_SHAPE, 
        n_classes=len(label_map), 
        augment=True, # <--- Bật chế độ biến hình
        shuffle=True
    )

    # Val: KHÔNG Augment (Chỉ kiểm tra dữ liệu gốc)
    val_gen = DataGenerator(
        X_val, y_val, 
        batch_size=BATCH_SIZE, 
        dim=INPUT_SHAPE, 
        n_classes=len(label_map), 
        augment=False,
        shuffle=False
    )

    # 4. Xây dựng và Train Model
    model = build_model(len(label_map))
    
    callbacks = [
        # Lưu model có Val Accuracy cao nhất
        ModelCheckpoint(os.path.join(MODELS_DIR, 'best_model.keras'), 
                        save_best_only=True, 
                        monitor='val_accuracy', 
                        mode='max',
                        verbose=1),
        
        # Giảm LR nếu loss không giảm trong 5 epoch
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6),
        
        # Dừng sớm nếu không cải thiện trong 20 epoch
        EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1)
    ]
    
    print("Start Training on GPU...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("Training Finished")