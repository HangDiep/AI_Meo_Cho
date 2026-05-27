"""
predict.py — Dự đoán khẩu trang trên một ảnh đơn lẻ.

Cách chạy:
    cd d:/AI/Khuau_trang
    python src/predict.py path/to/image.jpg
"""

import os
import sys
import io

# Cấu hình encoding UTF-8 tránh UnicodeEncodeError trên Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src.config import (
    BEST_MODEL_FINAL, BEST_MODEL_PHASE1,
    IMG_SIZE, RESCALE, CLASS_NAMES, DETECTION_CONFIDENCE,
)


def predict_single(image_path, model=None):
    """
    Dự đoán nhãn cho 1 ảnh.

    Args:
        image_path: Đường dẫn tới ảnh.
        model: Model đã load (nếu None sẽ tự load).

    Returns:
        tuple: (class_name, confidence)
    """
    # Load model nếu chưa có
    if model is None:
        if not os.path.exists(BEST_MODEL_FINAL):
            raise FileNotFoundError(f"Không tìm thấy model tại {BEST_MODEL_FINAL}.")
        model = tf.keras.models.load_model(
            BEST_MODEL_FINAL,
            compile=False
        )

    # Load và tiền xử lý ảnh
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) * RESCALE
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

    # Dự đoán
    # Dự đoán
    predictions = model.predict(img_array, verbose=0)[0]

    pred_class = np.argmax(predictions)
    confidence = predictions[pred_class]

    class_name = CLASS_NAMES[pred_class]

    return class_name, confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng: python src/predict.py <đường_dẫn_ảnh>")
        print("Ví dụ:     python src/predict.py Facemaskdataset/test/With_mask/0.png")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy file: {image_path}")
        sys.exit(1)

    class_name, confidence = predict_single(image_path)
    print(f"\n🎯 Kết quả: {class_name}")
    print(f"   Confidence: {confidence:.2%}")
