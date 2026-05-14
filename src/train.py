"""
train.py — Script huấn luyện mô hình 2 pha.

Pha 1: Freeze base model, chỉ train classification head.
Pha 2: Unfreeze một phần base model, fine-tune toàn bộ.

Cách chạy:
    cd d:/AI/Khuau_trang
    python src/train.py
"""

import os
import sys
import numpy as np

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from src.config import (
    PHASE1_EPOCHS, PHASE1_LEARNING_RATE,
    PHASE2_EPOCHS, PHASE2_LEARNING_RATE, FINE_TUNE_AT,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, MIN_LEARNING_RATE,
    BEST_MODEL_PHASE1, BEST_MODEL_FINAL, TRAINING_HISTORY,
    LOGS_DIR, PLOTS_DIR,
)
from src.model import build_model, unfreeze_model
from src.data_preprocessing import create_data_generators
from src.utils import plot_training_history


def get_callbacks(model_save_path, log_subdir="phase1"):
    """Tạo danh sách callbacks cho training."""
    return [
        # Dừng sớm nếu val_loss không giảm
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        # Lưu model tốt nhất
        ModelCheckpoint(
            model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # Giảm learning rate khi val_loss bão hòa
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=1,
        ),
        # TensorBoard logs
        TensorBoard(
            log_dir=os.path.join(LOGS_DIR, log_subdir),
            histogram_freq=0,
        ),
    ]


def train():
    """Chạy toàn bộ quy trình huấn luyện 2 pha."""

    # =================================================================
    # KIỂM TRA GPU
    # =================================================================
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"\n🚀 Phát hiện {len(gpus)} GPU: {gpus}")
        # Tránh lỗi OOM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("\n⚠️  Không có GPU — Sử dụng CPU (sẽ chậm hơn)")

    # =================================================================
    # TẠO DATA GENERATORS
    # =================================================================
    print("\n📦 Tạo data generators...")
    train_gen, val_gen, test_gen = create_data_generators()

    # =================================================================
    # PHA 1: TRAIN CLASSIFICATION HEAD
    # =================================================================
    print("\n" + "=" * 60)
    print("  PHA 1: TRAIN CLASSIFICATION HEAD (Base frozen)")
    print("=" * 60)

    model = build_model(
        learning_rate=PHASE1_LEARNING_RATE,
        freeze_base=True,
    )

    callbacks_p1 = get_callbacks(BEST_MODEL_PHASE1, "phase1")

    history_p1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE1_EPOCHS,
        callbacks=callbacks_p1,
    )

    # Lưu biểu đồ pha 1
    plot_training_history(history_p1, PLOTS_DIR)

    print(f"\n✅ Pha 1 hoàn tất! Best model lưu tại: {BEST_MODEL_PHASE1}")

    # =================================================================
    # PHA 2: FINE-TUNE
    # =================================================================
    print("\n" + "=" * 60)
    print("  PHA 2: FINE-TUNING (Unfreeze base model)")
    print("=" * 60)

    # Load best model từ pha 1
    model = tf.keras.models.load_model(BEST_MODEL_PHASE1)

    # Unfreeze một phần base model
    model = unfreeze_model(model, FINE_TUNE_AT, PHASE2_LEARNING_RATE)

    callbacks_p2 = get_callbacks(BEST_MODEL_FINAL, "phase2")

    history_p2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE2_EPOCHS,
        callbacks=callbacks_p2,
    )

    # =================================================================
    # LƯU KẾT QUẢ
    # =================================================================

    # Gộp history 2 pha
    combined_history = {}
    for key in history_p1.history:
        combined_history[key] = history_p1.history[key] + history_p2.history[key]

    # Lưu history
    np.save(TRAINING_HISTORY, combined_history)
    print(f"[OK] Đã lưu training history: {TRAINING_HISTORY}")

    # Vẽ biểu đồ gộp
    plot_training_history(combined_history, PLOTS_DIR)

    # =================================================================
    # ĐÁNH GIÁ NHANH TRÊN TẬP TEST
    # =================================================================
    print("\n📊 Đánh giá nhanh trên tập test...")
    loss, accuracy = model.evaluate(test_gen)
    print(f"\n  Test Loss:     {loss:.4f}")
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print(f"\n✅ HUẤN LUYỆN HOÀN TẤT!")
    print(f"   Model tốt nhất: {BEST_MODEL_FINAL}")
    print(f"   Biểu đồ:        {PLOTS_DIR}")
    print(f"   Logs:            {LOGS_DIR}")
    print(f"\n💡 Tiếp theo: chạy 'python src/evaluate.py' để đánh giá chi tiết.")


if __name__ == "__main__":
    train()
