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

DATA_DIR = 'Prepared_Data_All' 
LABEL_MAP_PATH = 'label_map.json'
MODELS_DIR = 'Models'


EPOCHS = 200       
BATCH_SIZE = 32         
INPUT_SHAPE = (60, 225) 
LEARNING_RATE = 1e-3  


def augment_data(data):
  
  
    aug_data = data.copy()
    

    scale = np.random.uniform(0.85, 1.15)
    aug_data = aug_data * scale

    shift_x = np.random.uniform(-0.1, 0.1)
    shift_y = np.random.uniform(-0.1, 0.1)
   
    aug_data[:, 0::3] += shift_x
    aug_data[:, 1::3] += shift_y 

    if np.random.rand() > 0.5:
        
        new_len = int(np.random.uniform(50, 70))
      
        aug_data = cv2.resize(aug_data, (225, new_len))
        
        
        if new_len < 60:
   
            pad = np.zeros((60 - new_len, 225))
            aug_data = np.vstack([aug_data, pad])
        else:
        
            aug_data = aug_data[:60, :]

  
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
            
      
            if data.shape != self.dim:
       
                data = cv2.resize(data, (self.dim[1], self.dim[0]))

            X[i,] = data
            y[i] = self.labels[k]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)


def build_model(num_classes):
    model = Sequential()
    
  
    model.add(Masking(mask_value=0.0, input_shape=INPUT_SHAPE))
  
    model.add(GRU(256, return_sequences=True, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4)) 
    
    model.add(GRU(128, return_sequences=False, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
   
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    
    
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    
   
    model.compile(
        optimizer=optimizer, 
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), 
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    
  
    if not os.path.exists(LABEL_MAP_PATH):
        print(f"Lỗi: Không tìm thấy {LABEL_MAP_PATH}. Hãy chạy prepare_data.py trước!")
        exit()

    with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    print(f"Loaded {len(label_map)} classes.")

  
    print("Scanning files...")
    files = glob.glob(os.path.join(DATA_DIR, '**', '*.npy'), recursive=True)
    

    label_dict = defaultdict(list)
    for f in files:
       
        label_name = os.path.basename(os.path.dirname(f))
        if label_name in label_map:
            label_dict[label_name].append(f)

    X_train, y_train = [], []
    X_val, y_val = [], []

    print("Splitting data (One-Shot Logic)...")
    for label_name, paths in label_dict.items():
        lbl_idx = label_map[label_name]
        

        if len(paths) == 1:
       
            X_train.extend(paths)
            y_train.append(lbl_idx)
            
            X_val.extend(paths)
            y_val.append(lbl_idx)
        else:
          
            split = int(len(paths) * 0.8)
            if split == 0: split = 1 
            
            X_train.extend(paths[:split])
            y_train.extend([lbl_idx]*len(paths[:split]))
          
            val_part = paths[split:]
            if len(val_part) == 0: val_part = [paths[-1]]
            
            X_val.extend(val_part)
            y_val.extend([lbl_idx]*len(val_part))

    print(f"--> Train set: {len(X_train)} files")
    print(f"--> Val set:   {len(X_val)} files")

    train_gen = DataGenerator(
        X_train, y_train, 
        batch_size=BATCH_SIZE, 
        dim=INPUT_SHAPE, 
        n_classes=len(label_map), 
        augment=True,
        shuffle=True
    )


    val_gen = DataGenerator(
        X_val, y_val, 
        batch_size=BATCH_SIZE, 
        dim=INPUT_SHAPE, 
        n_classes=len(label_map), 
        augment=False,
        shuffle=False
    )

   
    model = build_model(len(label_map))
    
    callbacks = [
      
        ModelCheckpoint(os.path.join(MODELS_DIR, 'best_model.keras'), 
                        save_best_only=True, 
                        monitor='val_accuracy', 
                        mode='max',
                        verbose=1),
        
        
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6),
        
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