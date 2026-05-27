"""
train.py — Script huấn luyện mô hình 2 pha.

Pha 1: Freeze base model, chỉ train classification head.
Pha 2: Unfreeze một phần base model để fine-tune.

Cách chạy:
    python src/train.py
"""

import os
import sys
import shutil
import numpy as np
import tensorflow as tf

# ============================================================
# THÊM ROOT PROJECT VÀO PATH
# ============================================================

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

# ============================================================
# CALLBACKS
# ============================================================

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# ============================================================
# IMPORT CONFIG
# ============================================================

from src.config import (
    PHASE1_EPOCHS,
    PHASE1_LEARNING_RATE,
    PHASE2_EPOCHS,
    PHASE2_LEARNING_RATE,
    FINE_TUNE_AT,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    MIN_LEARNING_RATE,
    BEST_MODEL_PHASE1,
    BEST_MODEL_FINAL,
    TRAINING_HISTORY,
    LOGS_DIR,
    PLOTS_DIR,
)

# ============================================================
# IMPORT MODULES
# ============================================================

from src.model import build_model, unfreeze_model
from src.data_preprocessing import create_data_generators
from src.utils import plot_training_history


# ============================================================
# CALLBACKS
# ============================================================

def get_callbacks(model_save_path):
    """
    Tạo callbacks cho training.
    """

    return [

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),

        # ====================================================
        # REDUCE LR
        # ====================================================

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=1,
        ),
    ]


# ============================================================
# TRAIN
# ============================================================

def train():

    print("\n" + "=" * 60)
    print(" KHỞI TẠO TRAINING ")
    print("=" * 60)

    # ========================================================
    # XÓA LOGS CŨ
    # ========================================================

    if os.path.exists(LOGS_DIR):
        shutil.rmtree(LOGS_DIR)

    os.makedirs(LOGS_DIR, exist_ok=True)

    # ========================================================
    # GPU CHECK
    # ========================================================

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:

        print(f"\n🚀 Phát hiện {len(gpus)} GPU")

        for gpu in gpus:
            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )

        policy = tf.keras.mixed_precision.Policy(
            "mixed_float16"
        )

        tf.keras.mixed_precision.set_global_policy(
            policy
        )

        print("⚡ Mixed precision enabled")

    else:

        print("\n⚠️ Không có GPU — dùng CPU")

    # ========================================================
    # DATA GENERATORS
    # ========================================================

    print("\n📦 Tạo data generators...")

    train_gen, val_gen, test_gen = (
        create_data_generators()
    )

    # ========================================================
    # PHA 1
    # ========================================================

    print("\n" + "=" * 60)
    print(" PHA 1 — TRAIN CLASSIFICATION HEAD ")
    print("=" * 60)

    model = build_model(
        learning_rate=PHASE1_LEARNING_RATE,
        freeze_base=True, #V ĐÓNG
    )

    callbacks_p1 = get_callbacks(
        BEST_MODEL_PHASE1
    )

    history_p1 = model.fit(

        train_gen,

        validation_data=val_gen,

        epochs=PHASE1_EPOCHS,

        callbacks=callbacks_p1,

        class_weight={
            0: 2.0,
            1: 1.0,
            2: 1.0,
        },
    )

    print("\n✅ Pha 1 hoàn tất")

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    model = tf.keras.models.load_model(
        BEST_MODEL_PHASE1
    )

    # ========================================================
    # PHA 2
    # ========================================================

    print("\n" + "=" * 60)
    print(" PHA 2 — FINE TUNING ")
    print("=" * 60)

    model = unfreeze_model(
        model,
        FINE_TUNE_AT,
        PHASE2_LEARNING_RATE,
    )

    callbacks_p2 = get_callbacks(
        BEST_MODEL_FINAL
    )

    history_p2 = model.fit(

        train_gen,

        validation_data=val_gen,

        epochs=PHASE2_EPOCHS,

        callbacks=callbacks_p2,

        class_weight={
            0: 1.7,
            1: 1.0,
            2: 1.0,
        },
    )

    print("\n✅ Pha 2 hoàn tất")

    # ========================================================
    # GỘP HISTORY
    # ========================================================

    combined_history = {}

    for key in history_p1.history:

        combined_history[key] = (

            history_p1.history[key]
            +
            history_p2.history[key]
        )

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    np.save(
        TRAINING_HISTORY,
        combined_history
    )

    print(
        f"\n💾 Đã lưu history: {TRAINING_HISTORY}"
    )

    # ========================================================
    # VẼ BIỂU ĐỒ
    # ========================================================

    plot_training_history(
        combined_history,
        PLOTS_DIR
    )

    # ========================================================
    # EVALUATE
    # ========================================================

    print("\n📊 Đánh giá trên tập test...")

    loss, accuracy = model.evaluate(
        test_gen
    )

    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # ========================================================
    # DONE
    # ========================================================

    print("\n" + "=" * 60)
    print(" ✅ HUẤN LUYỆN HOÀN TẤT ")
    print("=" * 60)

    print(f"\n📁 Model tốt nhất:")
    print(BEST_MODEL_FINAL)

    print(f"\n📁 History:")
    print(TRAINING_HISTORY)

    print(f"\n📁 Plots:")
    print(PLOTS_DIR)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train()