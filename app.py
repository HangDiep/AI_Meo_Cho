import base64
import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from tensorflow.keras.preprocessing.image import img_to_array

import src.realtime_detect as realtime_detect
from src.config import (
    BEST_MODEL_FINAL,
    BEST_MODEL_PHASE1,
    HAARCASCADE_PATH,
    HAARCASCADE_DIR,
    IMG_SIZE,
    RESCALE,
    CLASS_NAMES,
    DETECTION_CONFIDENCE,
)

app = Flask(__name__, static_folder="frontend", static_url_path="")


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


def load_model():
    model_path = BEST_MODEL_FINAL if os.path.exists(BEST_MODEL_FINAL) else BEST_MODEL_PHASE1
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError("Không tìm thấy model. Hãy chạy train và tạo file model trong thư mục models.")

    return tf.keras.models.load_model(model_path, compile=False)


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


def predict_frame(frame, model, face_cascade):
    if frame is None:
        return {"status": "error", "message": "Ảnh không hợp lệ"}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        return {"status": "no_face", "message": "Không tìm thấy khuôn mặt trên camera"}

    x, y, w, h = faces[0]
    face_roi = frame[y : y + h, x : x + w]
    face_resized = cv2.resize(face_roi, IMG_SIZE)
    face_array = img_to_array(face_resized) * RESCALE
    face_input = np.expand_dims(face_array, axis=0)

    prob = float(model.predict(face_input, verbose=0)[0][0])
    if prob >= DETECTION_CONFIDENCE:
        label = CLASS_NAMES[1]
        confidence = prob
    else:
        label = CLASS_NAMES[0]
        confidence = 1.0 - prob

    return {
        "status": "ok",
        "label": label,
        "confidence": confidence,
        "probability": prob,
    }


@app.route("/")
def index():
    return send_from_directory("frontend", "frontend_1.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        stream_with_context(
            realtime_detect.generate_frames(
                app.config["MODEL"],
                app.config["FACE_CASCADE"],
                use_mediapipe=app.config.get("USE_MEDIAPIPE", False),
                face_detector=app.config.get("FACE_DETECTOR", None),
            )
        ),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True)
    if not payload or "image" not in payload:
        return jsonify({"status": "error", "message": "Thiếu trường image"}), 400

    frame = decode_image(payload["image"])
    if frame is None:
        return jsonify({"status": "error", "message": "Không giải mã được ảnh"}), 400

    result = predict_frame(frame, app.config["MODEL"], app.config["FACE_CASCADE"])
    return jsonify(result)


if __name__ == "__main__":
    model, face_cascade, use_mediapipe, face_detector = realtime_detect.init_realtime_resources()
    app.config["MODEL"] = model
    app.config["FACE_CASCADE"] = face_cascade
    app.config["USE_MEDIAPIPE"] = use_mediapipe
    app.config["FACE_DETECTOR"] = face_detector

    app.run(host="0.0.0.0", port=5000, debug=True)
