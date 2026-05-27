import os
import sys
import cv2
import numpy as np
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

from src.config import (
    BEST_MODEL_FINAL,
    HAARCASCADE_PATH,
    HAARCASCADE_DIR,
    IMG_SIZE,
    RESCALE,
    CLASS_NAMES,
    WEBCAM_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    COLOR_MASK,
    COLOR_NO_MASK,
    COLOR_INCORRECT_MASK,
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
# DOWNLOAD HAAR
# =========================================================
def download_haarcascade():

    if os.path.exists(HAARCASCADE_PATH):
        return True

    cv2_data = os.path.join(os.path.dirname(cv2.__file__), "data")
    cascade_src = os.path.join(cv2_data, "haarcascade_frontalface_default.xml")

    if os.path.exists(cascade_src):

        import shutil
        os.makedirs(HAARCASCADE_DIR, exist_ok=True)
        shutil.copy2(cascade_src, HAARCASCADE_PATH)
        return True

    return False


# =========================================================
# INIT
# =========================================================
def init_realtime_resources():

    if not download_haarcascade():
        raise RuntimeError("Missing Haar Cascade")

    face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)

    print("[OK] Haar loaded")

    if not os.path.exists(BEST_MODEL_FINAL):
        raise FileNotFoundError(BEST_MODEL_FINAL)

    print("[OK] Loading model...")

    model = tf.keras.models.load_model(BEST_MODEL_FINAL, compile=False)

    print("[OK] Model loaded")

    return model, face_cascade, False, None


# =========================================================
# CORE PROCESS (USED BY BOTH MODES)
# =========================================================
def process_frame(frame, model, face_cascade):

    global _face_buffers, _last_preds, _frame_count

    _frame_count += 1
    should_predict = (_frame_count % PREDICT_EVERY_N_FRAMES == 0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        1.1,
        5,
        minSize=(60, 60)
    )

    faces = list(faces) if len(faces) > 0 else []
    faces = sorted(faces, key=lambda f: f[0])

    mask_count = no_mask_count = incorrect_count = 0
    active_slots = set()

    for slot_idx, (x, y, w, h) in enumerate(faces):

        active_slots.add(slot_idx)

        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            continue

        # ===================== PREDICT =====================
        if should_predict:

            face_r = cv2.resize(face, IMG_SIZE)
            face_r = cv2.cvtColor(face_r, cv2.COLOR_BGR2RGB)

            face_r = img_to_array(face_r) * RESCALE
            face_r = np.expand_dims(face_r, axis=0)

            preds = model.predict(face_r, verbose=0)[0]

            class_id = int(np.argmax(preds))
            confidence = float(preds[class_id])

            if confidence >= CONF_THRESHOLD:

                if slot_idx not in _face_buffers:
                    _face_buffers[slot_idx] = deque(maxlen=5)

                _face_buffers[slot_idx].append(class_id)
                _last_preds[slot_idx] = (class_id, confidence)

        # ===================== WAITING =====================
        if slot_idx not in _face_buffers:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (180,180,180), 2)
            cv2.putText(frame, "Detecting...", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 2)
            continue

        buffer = _face_buffers[slot_idx]
        final_class = max(set(buffer), key=buffer.count)

        _, conf = _last_preds.get(slot_idx, (final_class, 0.0))
        label = CLASS_NAMES[final_class]

        # ===================== COLOR =====================
        if label == "With_mask":
            color = COLOR_MASK
            mask_count += 1
        elif label == "Without_mask":
            color = COLOR_NO_MASK
            no_mask_count += 1
        else:
            color = COLOR_INCORRECT_MASK
            incorrect_count += 1

        # ===================== DRAW =====================
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.putText(
            frame,
            f"{label} {conf:.0%}",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # cleanup
    for k in list(_face_buffers.keys()):
        if k not in active_slots:
            del _face_buffers[k]
            _last_preds.pop(k, None)

    return frame


# =========================================================
# 🔥 FLASK REQUIRED FUNCTION
# =========================================================
def generate_frames(model, face_cascade, use_mediapipe=False, face_detector=None):

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    while True:

        success, frame = cap.read()
        if not success:
            break

        frame = process_frame(frame, model, face_cascade)

        _, buffer = cv2.imencode(".jpg", frame)

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )


# =========================================================
# DESKTOP MODE
# =========================================================
def run_realtime():

    model, face_cascade, _, _ = init_realtime_resources()

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, model, face_cascade)

        cv2.imshow("Mask Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    run_realtime()