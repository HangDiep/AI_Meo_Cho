import os
import sys
import cv2
import numpy as np
from collections import deque

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

import mediapipe as mp

# =========================================================
# CONFIG
# =========================================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    BEST_MODEL_FINAL,
    BEST_MODEL_PHASE1,
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
# INIT
# =========================================================
def init_realtime_resources():

    # MediaPipe
    face_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=DETECTION_CONFIDENCE
    )

    print("[OK] MediaPipe initialized")

    # =====================================================
    # LOAD MODEL (FIX HERE)
    # =====================================================
    model = None

    model_candidates = [
        BEST_MODEL_FINAL,
        BEST_MODEL_PHASE1
    ]

    print("MODEL CHECK:", BEST_MODEL_FINAL)

    for model_path in model_candidates:

        if not os.path.exists(model_path):
            continue

        try:
            print(f"[*] Loading model: {model_path}")

            # 🔥 FIX QUAN TRỌNG: CHỈ DÙNG tf.keras + compile=False
            model = tf.keras.models.load_model(
                model_path,
                compile=False
            )

            print("[OK] Model loaded successfully")
            break

        except Exception as e:
            print("❌ MODEL LOAD ERROR:")
            import traceback
            traceback.print_exc()

    if model is None:
        raise RuntimeError("Cannot load model (file hoặc version TF/Keras mismatch)")

    return model, None, True, face_detector


# =========================================================
# PROCESS FRAME
# =========================================================
def process_frame(frame, model, face_cascade, use_mediapipe=False, face_detector=None):

    global _face_buffers, _last_preds, _frame_count

    _frame_count += 1
    should_predict = (_frame_count % PREDICT_EVERY_N_FRAMES == 0)

    faces = []

    # MediaPipe detect
    if use_mediapipe and face_detector is not None:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb)

        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box

                x = int(bbox.xmin * frame.shape[1])
                y = int(bbox.ymin * frame.shape[0])
                w = int(bbox.width * frame.shape[1])
                h = int(bbox.height * frame.shape[0])

                x, y = max(0, x), max(0, y)

                faces.append((x, y, w, h))

    faces = sorted(faces, key=lambda f: f[0])

    mask_count = 0
    no_mask_count = 0
    incorrect_count = 0

    active_slots = set()

    for slot_idx, (x, y, w, h) in enumerate(faces):

        active_slots.add(slot_idx)

        pad = 20
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        if should_predict:

            face = cv2.resize(face, IMG_SIZE)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            arr = img_to_array(face) * RESCALE
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
            mask_count += 1

        elif label == "Without_mask":
            color = COLOR_NO_MASK
            no_mask_count += 1

        else:
            color = COLOR_INCORRECT_MASK
            incorrect_count += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label} {conf:.0%}",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2)

    return frame


# =========================================================
# RUN
# =========================================================
def run_realtime():

    model, _, use_mediapipe, face_detector = init_realtime_resources()

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("Running realtime... Press Q to quit")

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(
            frame,
            model,
            None,
            use_mediapipe,
            face_detector
        )

        cv2.imshow("Mask Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime()