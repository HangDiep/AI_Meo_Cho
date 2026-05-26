"""
config.py — Cấu hình tập trung cho toàn bộ dự án Mask Detection.
Thay đổi cấu hình tại đây thay vì sửa trực tiếp trong từng file.
"""

import os

# =============================================================================
# ĐƯỜNG DẪN
# =============================================================================

# Thư mục gốc dự án (tự động xác định từ vị trí file config.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset
DATASET_DIR = os.path.join(BASE_DIR, "Facemaskdataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Output
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
LOGS_DIR = os.path.join(BASE_DIR, "outputs", "logs")

# Haar Cascade (dùng cho realtime face detection)
HAARCASCADE_DIR = os.path.join(BASE_DIR, "haarcascade")
HAARCASCADE_PATH = os.path.join(HAARCASCADE_DIR, "haarcascade_frontalface_default.xml")

# =============================================================================
# CLASSES
# =============================================================================

CLASS_NAMES = [
    "incorrect_mask",
    "With_mask",
    "Without_mask"
]  # Thứ tự phải khớp với thư mục con trong dataset
# Label mapping: 0 = With_mask, 1 = Without_mask (theo thứ tự thư mục)

# =============================================================================
# TIỀN XỬ LÝ ẢNH
# =============================================================================

IMG_SIZE = (224, 224)       # Kích thước input cho 
IMG_SHAPE = (224, 224, 3)   # Shape đầy đủ (width, height, channels)
BATCH_SIZE = 32
RESCALE = 1.0 / 255        # Chuẩn hóa pixel [0, 255] → [0, 1]

# =============================================================================
# DATA AUGMENTATION (chỉ áp dụng cho tập train)
# =============================================================================

AUGMENTATION = {
    "rotation_range": 45,  # Xoay ngẫu nhiên trong khoảng ±45 độ
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "zoom_range": 0.3,
    "shear_range": 0.2,
    "brightness_range": [0.6, 1.3],
    "horizontal_flip": True,
    "fill_mode": "nearest"
}

# =============================================================================
# MÔ HÌNH
# =============================================================================

# Transfer Learning base
BASE_MODEL_NAME = "MobileNetV2"  # Nâng cấp từ MobileNetV2
BASE_MODEL_WEIGHTS = "imagenet"
FREEZE_BASE = True  # Freeze base model ở pha 1

# Classification Head
DENSE_UNITS = 128
DROPOUT_RATE_1 = 0.5    # Sau GlobalAveragePooling2D
DROPOUT_RATE_2 = 0.3    # Sau Dense layer

# =============================================================================
# HUẤN LUYỆN — PHA 1 (Freeze base)
# =============================================================================

PHASE1_EPOCHS = 14  # Giảm từ 15 để tối ưu thời gian
PHASE1_LEARNING_RATE = 1e-4  # 0.0001

# =============================================================================
# HUẤN LUYỆN — PHA 2 (Fine-tune)
# =============================================================================

PHASE2_EPOCHS = 18  # Giảm từ 20 để tối ưu thời gian
PHASE2_LEARNING_RATE = 1e-5   # 0.00005 (tăng từ 1e-5)
FINE_TUNE_AT = 130           # Unfreeze từ layer thứ 100 trở đi (MobileNetV2 có 155 layers)

# =============================================================================
# CALLBACKS
# =============================================================================

EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-7

# =============================================================================
# MODEL SAVE PATHS
# =============================================================================

BEST_MODEL_PHASE1 = os.path.join(MODELS_DIR, "best_phase1.keras")
BEST_MODEL_FINAL = os.path.join(MODELS_DIR, "best_final.keras")
TRAINING_HISTORY = os.path.join(MODELS_DIR, "training_history.npy")

# =============================================================================
# REALTIME DETECTION
# =============================================================================

WEBCAM_INDEX = 0            # 0 = webcam mặc định
DETECTION_CONFIDENCE = 0.5  # Ngưỡng tin cậy để hiển thị label
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Màu bounding box (BGR format cho OpenCV)
COLOR_MASK = (0, 255, 0)        # Xanh lá — Có khẩu trang
COLOR_NO_MASK = (0, 0, 255)     # Đỏ — Không khẩu trang
COLOR_INCORRECT_MASK = (0, 255, 255)  # Vàng — Đeo sai cách
FONT = None  # Sẽ dùng cv2.FONT_HERSHEY_SIMPLEX trong code

# =============================================================================
# TẠO THƯ MỤC NẾU CHƯA TỒN TẠI
# =============================================================================

for dir_path in [MODELS_DIR, PLOTS_DIR, REPORTS_DIR, LOGS_DIR, HAARCASCADE_DIR]:
    os.makedirs(dir_path, exist_ok=True)
