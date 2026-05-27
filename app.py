import base64
import os
import sys
import io

import cv2
import numpy as np
import tensorflow as tf

from flask import (
    Flask,
    Response,
    jsonify,
    request,
    send_from_directory,
    stream_with_context,
)

from src.config import (
    BEST_MODEL_FINAL,
    HAARCASCADE_PATH,
    HAARCASCADE_DIR,
    IMG_SIZE,
    RESCALE,
    CLASS_NAMES,
    DETECTION_CONFIDENCE,
)

from src.realtime_detect import (
    generate_frames,
    init_realtime_resources,
    GLOBAL_STATS
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    static_folder="frontend",
    static_url_path=""
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    if not os.path.exists(BEST_MODEL_FINAL):
        raise FileNotFoundError(f"Không tìm thấy model: {BEST_MODEL_FINAL}")

    model = tf.keras.models.load_model(BEST_MODEL_FINAL, compile=False)
    return model


# ============================================================
# DECODE BASE64 IMAGE
# ============================================================

def decode_image(data_url):
    if data_url.startswith("data:"):
        _, data_url = data_url.split(",", 1)

    try:
        image_data = base64.b64decode(data_url)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/about.html")
def about():
    return send_from_directory("frontend", "about.html")


# ============================================================
# VIDEO STREAM ROUTE (FIX CHÍNH)
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(
        stream_with_context(
            generate_frames(
                app.config["MODEL"],
                app.config["FACE_CASCADE"],
                use_mediapipe=app.config.get("USE_MEDIAPIPE", False),
                face_detector=app.config.get("FACE_DETECTOR", None),
            )
        ),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ============================================================
# API PREDICT
# ============================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():

    payload = request.get_json(force=True)

    if not payload or "image" not in payload:
        return jsonify({
            "status": "error",
            "message": "Thiếu image"
        }), 400

    frame = decode_image(payload["image"])

    if frame is None:
        return jsonify({
            "status": "error",
            "message": "Không decode được ảnh"
        }), 400

    # detect face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = app.config["FACE_CASCADE"].detectMultiScale(
        gray,
        1.1,
        5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return jsonify({
            "status": "no_face",
            "message": "Không tìm thấy khuôn mặt"
        })

    x, y, w, h = faces[0]
    face = frame[y:y+h, x:x+w]

    face = cv2.resize(face, IMG_SIZE)
    face = face.astype("float32") * RESCALE
    face = np.expand_dims(face, axis=0)

    preds = app.config["MODEL"].predict(face, verbose=0)[0]

    class_id = int(np.argmax(preds))
    confidence = float(preds[class_id])
    label = CLASS_NAMES[class_id]

    if confidence < DETECTION_CONFIDENCE:
        label = "Không chắc chắn"

    return jsonify({
        "status": "success",
        "label": label,
        "confidence": confidence
    })

# ============================================================
# STATS API
# ============================================================

@app.route("/api/stats")
def api_stats():

    return jsonify({
        "with_mask": GLOBAL_STATS["with_mask"],
        "without_mask": GLOBAL_STATS["without_mask"],
        "incorrect_mask": GLOBAL_STATS["incorrect_mask"],
        "total": GLOBAL_STATS["total"]
    })
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n🚀 Loading resources...")

    model = load_model()

    model, face_cascade, use_mediapipe, face_detector = init_realtime_resources()

    # override model
    model = load_model()

    app.config["MODEL"] = model
    app.config["FACE_CASCADE"] = face_cascade
    app.config["USE_MEDIAPIPE"] = use_mediapipe
    app.config["FACE_DETECTOR"] = face_detector

    print("\n✅ Server running at http://127.0.0.1:5000")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )