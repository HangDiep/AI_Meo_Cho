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
DATASET_DIR = os.path.join(BASE_DIR, "Facemaskdataset", "Facemaskdataset")
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



CLASS_NAMES = [
    "incorrect_mask",
    "With_mask",
    "Without_mask"
]  #===================

IMG_SIZE = (224, 224)      
IMG_SHAPE = (224, 224, 3)  
BATCH_SIZE = 32
RESCALE = 1.0 / 255       

AUGMENTATION = {
    "rotation_range": 45,        # Tăng từ 25 → 45 độ (cover góc nghiêng nhiều hơn)
    "width_shift_range": 0.3,
    "height_shift_range": 0.3,
    "shear_range": 0.3,          # Tăng từ 0.15 → 0.3 (capture biến dạng góc tốt hơn)
    "zoom_range": 0.2,           # Tăng từ 0.15 → 0.2
    "horizontal_flip": True,
    "brightness_range": [0.4, 2.0],  # Mở rộng dải sáng (từ 0.5-1.5 → 0.4-2.0)
    "fill_mode": "nearest"
}

BASE_MODEL_NAME = "MobileNetV2" 
BASE_MODEL_WEIGHTS = "imagenet"
FREEZE_BASE = True  

# Classification Head
DENSE_UNITS = 128
DROPOUT_RATE_1 = 0.5    
DROPOUT_RATE_2 = 0.3    


PHASE1_EPOCHS = 14  
PHASE1_LEARNING_RATE = 1e-4  #0.0001



PHASE2_EPOCHS = 18  
PHASE2_LEARNING_RATE = 1e-5   # 0.00005 (tăng từ 1e-5)
FINE_TUNE_AT = 130           

EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-7



BEST_MODEL_PHASE1 = os.path.join(MODELS_DIR, "best_phase1.keras")
BEST_MODEL_FINAL = os.path.join(MODELS_DIR, "best_final.keras")
TRAINING_HISTORY = os.path.join(MODELS_DIR, "training_history.npy")

# MediaPipe Tasks face detection model (MediaPipe >= 0.10)
MEDIAPIPE_FACE_MODEL = os.path.join(MODELS_DIR, "blaze_face_short_range.tflite")



WEBCAM_INDEX = 0           
DETECTION_CONFIDENCE = 0.6  # Ngưỡng tin cậy để hiển thị label
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


COLOR_MASK = (0, 255, 0)        # Xanh lá — Có khẩu trang
COLOR_NO_MASK = (0, 0, 255)     # Đỏ — Không khẩu trang
COLOR_INCORRECT_MASK = (0, 255, 255)  # Vàng — Đeo sai cách
FONT = None  # Sẽ dùng cv2.FONT_HERSHEY_SIMPLEX trong code

# =============================================================================
# TẠO THƯ MỤC NẾU CHƯA TỒN TẠI
# =============================================================================

for dir_path in [MODELS_DIR, PLOTS_DIR, REPORTS_DIR, LOGS_DIR, HAARCASCADE_DIR]:
    os.makedirs(dir_path, exist_ok=True)
print(TRAIN_DIR)
print(os.path.exists(TRAIN_DIR))