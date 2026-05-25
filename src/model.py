"""
model.py — Định nghĩa kiến trúc mô hình MobileNetV2 + Transfer Learning.

Module này cung cấp hàm build_model() để tạo model,
không chạy trực tiếp.
"""

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from src.config import (
    IMG_SHAPE, BASE_MODEL_WEIGHTS,
    DENSE_UNITS, DROPOUT_RATE_1, DROPOUT_RATE_2,
    PHASE1_LEARNING_RATE,
)


def build_model(learning_rate=None, freeze_base=True):
    """
    Xây dựng model MobileNetV2 + Classification Head.

    Args:
        learning_rate: Learning rate cho optimizer. Mặc định dùng từ config.
        freeze_base: True = freeze toàn bộ base model (pha 1).

    Returns:
        tf.keras.Model: Model đã compile, sẵn sàng train.
    """
    if learning_rate is None:
        learning_rate = PHASE1_LEARNING_RATE

    # ----- Base Model: EfficientNetB1 pretrained trên ImageNet -----
    base_model = EfficientNetB1(
        weights=BASE_MODEL_WEIGHTS,
        include_top=False,          # Bỏ classification head gốc
        input_shape=IMG_SHAPE,
    )

    # Freeze hoặc unfreeze base model
    base_model.trainable = not freeze_base

    # ----- Classification Head -----
    x = base_model.output
    x = GlobalAveragePooling2D()(x)     # Giảm chiều từ (7,7,1280) → (1280,)
    x = Dropout(DROPOUT_RATE_1)(x)      # Chống overfitting
    x = Dense(DENSE_UNITS, activation="relu")(x)  # Feature extraction
    x = Dropout(DROPOUT_RATE_2)(x)      # Thêm regularization
    output = Dense(3, activation="softmax")(x)     # Output: 0, 1, hoặc 2 (3 classes)

    # ----- Tạo Model -----
    model = Model(inputs=base_model.input, outputs=output)

    # ----- Compile -----
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ----- Thông tin model -----
    total_params = model.count_params()
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable = total_params - trainable

    print("\n" + "=" * 60)
    print(f"  MODEL: EfficientNetB1 + Custom Head")
    print(f"  Base frozen: {freeze_base}")
    print(f"  Total params:       {total_params:,}")
    print(f"  Trainable params:   {trainable:,}")
    print(f"  Non-trainable:      {non_trainable:,}")
    print(f"  Learning rate:      {learning_rate}")
    print("=" * 60)

    return model


def unfreeze_model(model, fine_tune_at, learning_rate):
    """
    Unfreeze một phần base model để fine-tune (pha 2).

    Args:
        model: Model đã train ở pha 1.
        fine_tune_at: Unfreeze từ layer index này trở đi.
        learning_rate: Learning rate mới (thường rất nhỏ).

    Returns:
        Model đã recompile, sẵn sàng cho pha 2.
    """
    # Tìm base model (MobileNetV2) trong các layers
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is not None:
        # Trường hợp model giữ nguyên cấu trúc (sub-model MobileNetV2 tồn tại)
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
    else:
        # Trường hợp model bị phẳng hóa khi load từ 
        # → duyệt trực tiếp trên model.layers
        for layer in model.layers:
            layer.trainable = True
        for layer in model.layers[:fine_tune_at]:
            layer.trainable = False

    # Recompile với learning rate thấp hơn
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)

    print("\n" + "=" * 60)
    print(f"  FINE-TUNE MODE")
    total_layers = len(base_model.layers) if base_model is not None else len(model.layers)
    print(f"  Unfreeze từ layer {fine_tune_at} / {total_layers}")
    print(f"  Trainable params:   {trainable:,}")
    print(f"  Learning rate:      {learning_rate}")
    print("=" * 60)

    return model


if __name__ == "__main__":
    # Test build model
    model = build_model(freeze_base=True)
    model.summary()
