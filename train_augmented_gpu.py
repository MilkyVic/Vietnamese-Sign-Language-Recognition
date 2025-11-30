import os
import glob
import json
import logging

# Force TensorFlow to see only GPU 0
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.model_selection import train_test_split


DATA_PATH = 'Data_gpu'
LABEL_MAP_PATH = 'Logs/label_map.json'
LOG_DIR = 'Logs'
BATCH_SIZE = 32
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
AUTOTUNE = tf.data.AUTOTUNE
SEQUENCE_SHAPE = (60, 201)


def ensure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError('No GPU detected. Please ensure CUDA device 0 is available.')
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as exc:
        tf.get_logger().warning('Could not set memory growth: %s', exc)


def load_label_map():
    with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    return label_map


def discover_files():
    pattern = os.path.join(DATA_PATH, '**', '*.npz')
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f'No .npz files found under {DATA_PATH}. Please run data augmentation first.')
    logging.info("Discovered %d .npz files under %s", len(files), DATA_PATH)
    return files


def stratify_keys(paths):
    # Use sanitized folder name (parent directory) for stratification
    return [os.path.basename(os.path.dirname(p)) for p in paths]


def split_dataset(all_files):
    strat_labels = stratify_keys(all_files)
    train_files, temp_files = train_test_split(
        all_files,
        test_size=VAL_SPLIT + TEST_SPLIT,
        shuffle=True,
        random_state=42,
        stratify=strat_labels if len(set(strat_labels)) > 1 else None,
    )

    if not temp_files:
        return train_files, [], []

    temp_strat = stratify_keys(temp_files)
    val_files, test_files = train_test_split(
        temp_files,
        test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT),
        shuffle=True,
        random_state=42,
        stratify=temp_strat if len(set(temp_strat)) > 1 else None,
    )

    return train_files, val_files, test_files


def _load_npz(path):
    npz_path = path.decode('utf-8')
    data = np.load(npz_path)
    seq = data['sequence'].astype(np.float32)
    lbl = np.int32(data['label'])
    return seq, lbl


def parse_fn(path):
    seq, lbl = tf.numpy_function(_load_npz, [path], [tf.float32, tf.int32])
    seq.set_shape(SEQUENCE_SHAPE)
    lbl.set_shape([])
    return seq, lbl


def make_dataset(file_list, shuffle=False, repeat=False):
    ds = tf.data.Dataset.from_tensor_slices(file_list)
    if shuffle:
        buffer = max(len(file_list), 1)
        ds = ds.shuffle(buffer, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    ds = ds.map(parse_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def build_model(num_classes):
    inputs = layers.Input(shape=SEQUENCE_SHAPE)

    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.3))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(256, dropout=0.3))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def create_callbacks():
    ckpt_dir = 'Models/checkpoints_gpu_2'
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_path = os.path.join(ckpt_dir, 'best_model_2.keras')
    csv_log_path = os.path.join(LOG_DIR, 'train_augmented_gpu_history.csv')

    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        CSVLogger(csv_log_path, append=True),
    ]
    return callbacks


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOG_DIR, 'train_augmented_gpu.log'), mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("Starting train_augmented_gpu pipeline.")
    ensure_gpu()
    label_map = load_label_map()
    num_classes = len(label_map)
    logging.info("Loaded %d classes.", num_classes)

    files = discover_files()
    train_files, val_files, test_files = split_dataset(files)

    logging.info("Train samples: %d", len(train_files))
    logging.info("  Val samples: %d", len(val_files))
    logging.info(" Test samples: %d", len(test_files))

    steps_per_epoch = max(len(train_files) // BATCH_SIZE, 1)
    validation_steps = max(len(val_files) // BATCH_SIZE, 1) if val_files else None

    train_ds = make_dataset(train_files, shuffle=True, repeat=True)
    val_ds = make_dataset(val_files, shuffle=False, repeat=False) if val_files else None

    model = build_model(num_classes)
    callbacks = create_callbacks()
    logging.info("Model compiled, starting training.")

    model.fit(
        train_ds,
        epochs=100,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    if test_files:
        test_ds = make_dataset(test_files, shuffle=False, repeat=False)
        test_steps = max(len(test_files) // BATCH_SIZE, 1)
        eval_results = model.evaluate(test_ds, steps=test_steps)
        metrics = dict(zip(model.metrics_names, eval_results))
        logging.info("Test results: %s", metrics)
    logging.info("Training completed.")


if __name__ == '__main__':
    main()
