import os
import sys
import io

# Cau hinh encoding UTF-8 tranh UnicodeEncodeError tren Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import cv2
import numpy as np
from collections import deque

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# MediaPipe Tasks API (>= 0.10)
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    BEST_MODEL_FINAL,
    MEDIAPIPE_FACE_MODEL,
    IMG_SIZE,
    RESCALE,
    CLASS_NAMES,
    WEBCAM_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    COLOR_MASK,
    COLOR_NO_MASK,
    COLOR_INCORRECT_MASK,
    DETECTION_CONFIDENCE,
)

# =========================================================
# GLOBAL STATE
# =========================================================
_face_buffers = {}
_last_preds = {}
_frame_count = 0

PREDICT_EVERY_N_FRAMES = 3
CONF_THRESHOLD = 0.7


# =========================================================
# INIT MEDIAPIPE FACE DETECTOR (Tasks API >= 0.10)
# =========================================================
def _init_mediapipe_detector():
    """Khoi tao MediaPipe FaceDetector dung Tasks API moi."""
    if not os.path.exists(MEDIAPIPE_FACE_MODEL):
        raise FileNotFoundError(
            f"Khong tim thay MediaPipe model: {MEDIAPIPE_FACE_MODEL}\n"
            "Chay lenh sau de tai ve:\n"
            "  python -c \"import urllib.request; urllib.request.urlretrieve("
            "'https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/1/blaze_face_short_range.tflite', "
            "'models/blaze_face_short_range.tflite')\""
        )

    base_options = mp_python.BaseOptions(model_asset_path=MEDIAPIPE_FACE_MODEL)
    options = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        min_detection_confidence=DETECTION_CONFIDENCE,
    )
    detector = mp_vision.FaceDetector.create_from_options(options)
    print("[OK] MediaPipe FaceDetector initialized (Tasks API)")
    return detector


# =========================================================
# DETECT FACES (MediaPipe Tasks API)
# =========================================================
def _detect_faces(frame, face_detector):
    """
    Phat hien khuon mat bang MediaPipe Tasks API.
    Tra ve list (x, y, w, h) theo pixel.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_detector.detect(mp_image)

    faces = []
    h_frame, w_frame = frame.shape[:2]

    for detection in result.detections:
        bbox = detection.bounding_box
        x = max(0, bbox.origin_x)
        y = max(0, bbox.origin_y)
        w = min(bbox.width, w_frame - x)
        h = min(bbox.height, h_frame - y)
        if w > 30 and h > 30:
            faces.append((x, y, w, h))

    return faces


# =========================================================
# INIT RESOURCES
# =========================================================
def init_realtime_resources():

    # ----- MediaPipe Face Detector -----
    face_detector = _init_mediapipe_detector()

    # ----- Load Keras model -----
    print(f"[*] Loading model: {BEST_MODEL_FINAL}")
    if not os.path.exists(BEST_MODEL_FINAL):
        raise FileNotFoundError(f"Khong tim thay model: {BEST_MODEL_FINAL}")

    try:
        model = tf.keras.models.load_model(BEST_MODEL_FINAL, compile=False)
        print("[OK] Model loaded successfully")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Cannot load model best_final.keras: {e}")

    # Tra ve: model, face_cascade=None, use_mediapipe=True, face_detector
    return model, None, True, face_detector


# =========================================================
# PROCESS FRAME
# =========================================================
def process_frame(frame, model, face_cascade, use_mediapipe=True, face_detector=None):

    global _face_buffers, _last_preds, _frame_count

    _frame_count += 1
    should_predict = (_frame_count % PREDICT_EVERY_N_FRAMES == 0)

    # Detect khuon mat
    if use_mediapipe and face_detector is not None:
        faces = _detect_faces(frame, face_detector)
    else:
        # Fallback: Haar Cascade neu khong co mediapipe
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        from src.config import HAARCASCADE_PATH
        cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
        detected = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        faces = list(detected) if len(detected) > 0 else []

    faces = sorted(faces, key=lambda f: f[0])
    active_slots = set()

    for slot_idx, (x, y, w, h) in enumerate(faces):

        active_slots.add(slot_idx)

        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        if should_predict:
            face_resized = cv2.resize(face, IMG_SIZE)
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            arr = img_to_array(face_rgb) * RESCALE
            arr = np.expand_dims(arr, axis=0)

            pred = model.predict(arr, verbose=0)[0]
            class_id = int(np.argmax(pred))
            confidence = float(pred[class_id])

            if confidence >= CONF_THRESHOLD:
                if slot_idx not in _face_buffers:
                    _face_buffers[slot_idx] = deque(maxlen=5)
                _face_buffers[slot_idx].append(class_id)
                _last_preds[slot_idx] = (class_id, confidence)

        if slot_idx not in _face_buffers:
            continue

        buffer = _face_buffers[slot_idx]
        final_class = max(set(buffer), key=buffer.count)
        _, conf = _last_preds.get(slot_idx, (final_class, 0.0))
        label = CLASS_NAMES[final_class]

        if label == "With_mask":
            color = COLOR_MASK
        elif label == "Without_mask":
            color = COLOR_NO_MASK
        else:
            color = COLOR_INCORRECT_MASK

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            f"{label} {conf:.0%}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    # Xoa slot khong con active
    for slot_idx in list(_face_buffers.keys()):
        if slot_idx not in active_slots:
            del _face_buffers[slot_idx]
            _last_preds.pop(slot_idx, None)

    return frame


# =========================================================
# GENERATE FRAMES (cho Flask video stream)
# =========================================================
def generate_frames(model, face_cascade, use_mediapipe=True, face_detector=None):

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_frame(frame, model, face_cascade, use_mediapipe, face_detector)

            _, buffer = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
    finally:
        cap.release()


# =========================================================
# RUN STANDALONE
# =========================================================
def run_realtime():

    model, face_cascade, use_mediapipe, face_detector = init_realtime_resources()

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("Running realtime... Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, model, face_cascade, use_mediapipe, face_detector)
        cv2.imshow("Mask Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime()