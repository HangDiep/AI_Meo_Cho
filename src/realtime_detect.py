"""
realtime_detect.py — Nhận diện khẩu trang realtime qua webcam.

Tính năng:
- Detect nhiều khuôn mặt cùng lúc
- Bounding box màu: Xanh (Mask) / Đỏ (No Mask)
- Hiển thị confidence %
- Thống kê realtime (tổng người, có mask, không mask)
- Phím tắt: Q = thoát, S = screenshot

Cách chạy:
    cd d:/AI/Khuau_trang
    python src/realtime_detect.py
"""

import os
import sys
import cv2
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

from src.config import (
    BEST_MODEL_FINAL, BEST_MODEL_PHASE1,
    HAARCASCADE_PATH, HAARCASCADE_DIR,
    IMG_SIZE, RESCALE, CLASS_NAMES, DETECTION_CONFIDENCE,
    WEBCAM_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    COLOR_MASK, COLOR_NO_MASK,
    PLOTS_DIR,
)


def download_haarcascade():
    """Download Haar Cascade XML nếu chưa có."""
    if os.path.exists(HAARCASCADE_PATH):
        return True

    print("📥 Đang tìm Haar Cascade...")

    # Thử lấy từ OpenCV đã cài
    cv2_data = os.path.join(os.path.dirname(cv2.__file__), "data")
    cascade_src = os.path.join(cv2_data, "haarcascade_frontalface_default.xml")

    if os.path.exists(cascade_src):
        import shutil
        os.makedirs(HAARCASCADE_DIR, exist_ok=True)
        shutil.copy2(cascade_src, HAARCASCADE_PATH)
        print(f"[OK] Copy Haar Cascade từ OpenCV: {HAARCASCADE_PATH}")
        return True

    print("❌ Không tìm thấy Haar Cascade!")
    print(f"   Tải thủ công từ:")
    print(f"   https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml")
    print(f"   Lưu vào: {HAARCASCADE_PATH}")
    return False


def run_realtime():
    """Chạy nhận diện khẩu trang realtime qua webcam."""

    # =================================================================
    # CHUẨN BỊ
    # =================================================================

    # Download/Copy Haar Cascade
    if not download_haarcascade():
        return

    # Load face detector
    face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
    if face_cascade.empty():
        print("⚠️ Không load được Haar Cascade, nhưng sẽ thử dùng MediaPipe trước.")

    # Initialize MediaPipe Face Detection
    try:
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        face_detector = mp_face_detection.FaceDetection(
            model_selection=0,  # 0: khoảng cách gần (webcam), 1: khoảng cách xa (5m)
            min_detection_confidence=0.5
        )
        use_mediapipe = True
        print("💡 Đang sử dụng bộ dò khuôn mặt MediaPipe (hỗ trợ góc nghiêng cực tốt).")
    except ImportError:
        use_mediapipe = False
        print("⚠️ Không thể import mediapipe. Tự động chuyển sang Haar Cascade (chỉ nhận diện mặt thẳng).")

    # Load mask detection model
    model_path = BEST_MODEL_FINAL if os.path.exists(BEST_MODEL_FINAL) else BEST_MODEL_PHASE1
    if not os.path.exists(model_path):
        print("❌ Không tìm thấy model! Hãy chạy train.py trước.")
        return

    print(f"📦 Loading model: {model_path}")
    model = tf.keras.models.load_model(
    model_path,
    compile=False
)

    # Mở webcam
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("❌ Không thể mở webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("\n🎥 WEBCAM REALTIME — Nhận diện khẩu trang")
    print("   Phím Q: Thoát")
    print("   Phím S: Chụp screenshot")
    print("=" * 50)

    screenshot_count = 0

    # =================================================================
    # VÒNG LẶP CHÍNH
    # =================================================================

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Không đọc được frame từ webcam")
            break

        # Detect khuôn mặt
        faces = []
        if use_mediapipe:
            # MediaPipe yêu cầu định dạng RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb_frame)
            
            ih, iw, _ = frame.shape
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)
                    
                    # Giới hạn bounding box trong khung hình để tránh lỗi index âm hoặc vượt quá kích thước ảnh
                    x_clipped = max(0, x)
                    y_clipped = max(0, y)
                    w_clipped = min(w + (x - x_clipped), iw - x_clipped)
                    h_clipped = min(h + (y - y_clipped), ih - y_clipped)
                    
                    if w_clipped > 30 and h_clipped > 30:  # Loại bỏ các vùng quá bé
                        faces.append((x_clipped, y_clipped, w_clipped, h_clipped))
        else:
            # Chuyển sang grayscale để detect khuôn mặt bằng Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )
            faces = list(detected_faces)

        # Thống kê
        total_faces = len(faces)
        mask_count = 0
        no_mask_count = 0

        # Xử lý từng khuôn mặt
        for (x, y, w, h) in faces:
            # Crop vùng mặt từ frame gốc (RGB)
            face_roi = frame[y:y + h, x:x + w]

            if face_roi.size == 0:
                continue

            # Tiền xử lý cho model
            face_resized = cv2.resize(face_roi, IMG_SIZE)
            face_array = img_to_array(face_resized) * RESCALE
            face_input = np.expand_dims(face_array, axis=0)

            # Dự đoán
            prob = model.predict(face_input, verbose=0)[0][0]

            if prob >= DETECTION_CONFIDENCE:
                label = CLASS_NAMES[1]  # Without_mask
                confidence = prob
                color = COLOR_NO_MASK
                no_mask_count += 1
            else:
                label = CLASS_NAMES[0]  # With_mask
                confidence = 1 - prob
                color = COLOR_MASK
                mask_count += 1

            # Vẽ bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Vẽ label + confidence
            text = f"{label}: {confidence:.0%}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            # Background cho text
            cv2.rectangle(
                frame,
                (x, y - text_size[1] - 10),
                (x + text_size[0], y),
                color, -1,
            )
            cv2.putText(
                frame, text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2,
            )

        # =============================================================
        # HIỂN THỊ THỐNG KÊ
        # =============================================================
        stats_y = 30
        cv2.putText(frame, f"Total: {total_faces}", (10, stats_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Mask: {mask_count}", (10, stats_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_MASK, 2)
        cv2.putText(frame, f"No Mask: {no_mask_count}", (10, stats_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_NO_MASK, 2)

        # Hiển thị frame
        cv2.imshow("Mask Detection — Press Q to quit", frame)

        # =============================================================
        # XỬ LÝ PHÍM
        # =============================================================
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord("s") or key == ord("S"):
            screenshot_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(PLOTS_DIR, f"screenshot_{timestamp}.png")
            cv2.imwrite(screenshot_path, frame)
            print(f"📸 Screenshot saved: {screenshot_path}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Đã tắt webcam.")


if __name__ == "__main__":
    run_realtime()
