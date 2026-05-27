import os
import sys
import cv2
import numpy as np
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =========================================================
# TENSORFLOW
# =========================================================
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# =========================================================
# CONFIG
# =========================================================
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
# SETTINGS
# =========================================================
_face_buffers = {}
_last_preds = {}
_frame_count = 0

PREDICT_EVERY_N_FRAMES = 3
CONF_THRESHOLD = 0.7


# =========================================================
# DOWNLOAD HAARCASCADE
# =========================================================
def download_haarcascade():

    if os.path.exists(HAARCASCADE_PATH):
        return True

    cv2_data = os.path.join(
        os.path.dirname(cv2.__file__),
        "data"
    )

    cascade_src = os.path.join(
        cv2_data,
        "haarcascade_frontalface_default.xml"
    )

    if os.path.exists(cascade_src):

        import shutil

        os.makedirs(
            HAARCASCADE_DIR,
            exist_ok=True
        )

        shutil.copy2(
            cascade_src,
            HAARCASCADE_PATH
        )

        return True

    return False


# =========================================================
# INIT
# =========================================================
def init_realtime_resources():

    # =========================
    # HAAR CASCADE
    # =========================
    if not download_haarcascade():
        raise RuntimeError("Missing Haar Cascade")

    face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)

    print("[OK] Using Haarcascade")

    # =========================
    # LOAD MODEL (ONLY 1 MODEL)
    # =========================
    if not os.path.exists(BEST_MODEL_FINAL):
        raise FileNotFoundError(f"Không tìm thấy model: {BEST_MODEL_FINAL}")

    try:
        print(f"[*] Loading model: {BEST_MODEL_FINAL}")

        model = tf.keras.models.load_model(
            BEST_MODEL_FINAL,
            compile=False
        )

        print("[OK] Model loaded successfully")

    except Exception as e:
        raise RuntimeError(f"Cannot load model: {e}")

    return model, face_cascade, False, None


# =========================================================
# PROCESS FRAME
# =========================================================
def process_frame(
    frame,
    model,
    face_cascade,
    use_mediapipe=False,
    face_detector=None
):

    global _face_buffers, _last_preds, _frame_count

    _frame_count += 1
    should_predict = (_frame_count % PREDICT_EVERY_N_FRAMES == 0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detected = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    faces = list(detected) if len(detected) > 0 else []
    faces = sorted(faces, key=lambda f: f[0])

    mask_count = 0
    no_mask_count = 0
    incorrect_count = 0

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

        # =================================================
        # PREDICT
        # =================================================
        if should_predict:

            face_resized = cv2.resize(face, IMG_SIZE)
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            face_array = img_to_array(face_rgb) * RESCALE
            face_input = np.expand_dims(face_array, axis=0)

            prediction = model.predict(face_input, verbose=0)[0]

            # ✅ FIX CHUẨN
            class_id = int(np.argmax(prediction))
            confidence = float(prediction[class_id])

            if confidence >= CONF_THRESHOLD:

                if slot_idx not in _face_buffers:
                    _face_buffers[slot_idx] = deque(maxlen=5)

                _face_buffers[slot_idx].append(class_id)
                _last_preds[slot_idx] = (class_id, confidence)

        # =================================================
        # NO DATA YET
        # =================================================
        if slot_idx not in _face_buffers or len(_face_buffers[slot_idx]) == 0:

            cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 180, 180), 2)
            cv2.putText(
                frame,
                "Detecting...",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 180, 180),
                2
            )
            continue

        # =================================================
        # SMOOTH RESULT
        # =================================================
        buffer = _face_buffers[slot_idx]
        final_class = max(set(buffer), key=buffer.count)

        _, last_conf = _last_preds.get(slot_idx, (final_class, 0.0))
        label = CLASS_NAMES[final_class]

        # =================================================
        # COLOR
        # =================================================
        if label == "With_mask":
            color = COLOR_MASK
            mask_count += 1

        elif label == "Without_mask":
            color = COLOR_NO_MASK
            no_mask_count += 1

        else:
            color = COLOR_INCORRECT_MASK
            incorrect_count += 1

        # =================================================
        # DRAW
        # =================================================
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.putText(
            frame,
            f"{label} {last_conf:.0%}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # =====================================================
    # CLEAN OLD BUFFER
    # =====================================================
    for slot in list(_face_buffers.keys()):
        if slot not in active_slots:
            del _face_buffers[slot]
            _last_preds.pop(slot, None)

    # =====================================================
    # STATS
    # =====================================================
    cv2.putText(frame, f"Total: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f"Mask: {mask_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_MASK, 2)

    cv2.putText(frame, f"No Mask: {no_mask_count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_NO_MASK, 2)

    cv2.putText(frame, f"Incorrect: {incorrect_count}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_INCORRECT_MASK, 2)

    return frame



# =========================================================
# GENERATE FRAMES FOR FLASK
# =========================================================
def generate_frames(
    model,
    face_cascade,
    use_mediapipe=False,
    face_detector=None
):

    cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():

        print("❌ Cannot open webcam")

        return

    print("✅ Webcam started")

    while True:

        success, frame = cap.read()

        if not success:

            print("❌ Cannot read frame")

            break

        try:

            # =============================================
            # PROCESS FRAME
            # =============================================

            frame = process_frame(

                frame,

                model,

                face_cascade,

                use_mediapipe,

                face_detector
            )

            # =============================================
            # ENCODE JPEG
            # =============================================

            ret, buffer = cv2.imencode(
                ".jpg",
                frame
            )

            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            # =============================================
            # STREAM FRAME
            # =============================================

            yield (

                b"--frame\r\n"

                b"Content-Type: image/jpeg\r\n\r\n"

                + frame_bytes +

                b"\r\n"
            )

        except Exception as e:

            print(f"❌ Frame Error: {e}")

            continue

    cap.release()
# =========================================================
# RUN REALTIME
# =========================================================
def run_realtime():

    model, face_cascade, _, _ = init_realtime_resources()

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("Running realtime... Press Q to quit")

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, model, face_cascade)

        cv2.imshow("Mask Detection Stable", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# =========================================================
# ENTRY
# =========================================================
if __name__ == "__main__":
    run_realtime()