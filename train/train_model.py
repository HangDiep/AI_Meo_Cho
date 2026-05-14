import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# =========================
# ĐƯỜNG DẪN DATASET
# =========================

train_dir = "../Facemaskdataset/train"
val_dir = "../Facemaskdataset/val"
test_dir = "../Facemaskdataset/test"

# Nếu dataset của bạn dùng "val"
# thì sửa:
# val_dir = "dataset/val"

# =========================
# GIAI ĐOẠN 2a + 2b
# Resize + Normalize + Augmentation
# =========================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,          # normalize pixel 0-255 -> 0-1
    rotation_range=20,          # xoay ảnh
    zoom_range=0.2,             # zoom
    horizontal_flip=True,       # lật ngang
    brightness_range=[0.8,1.2] # thay đổi độ sáng
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

# =========================
# GIAI ĐOẠN 2c
# Data Generator
# =========================

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# =========================
# GIAI ĐOẠN 3a
# MobileNetV2
# =========================

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# =========================
# THÊM CLASSIFICATION HEAD
# =========================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.5)(x)

x = Dense(128, activation='relu')(x)

output = Dense(1, activation='sigmoid')(x)

# =========================
# TẠO MODEL
# =========================

model = Model(
    inputs=base_model.input,
    outputs=output
)

# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# HIỂN THỊ MODEL
# =========================

model.summary()

# =========================
# TRAIN MODEL
# =========================

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

# =========================
# ĐÁNH GIÁ MODEL
# =========================

loss, accuracy = model.evaluate(test_generator)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# =========================
# LƯU MODEL
# =========================

model.save("models/mask_detector.h5")

print("Model saved!")