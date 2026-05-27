import base64
import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

from flask import (
    Flask,
    Response,
    jsonify,
    request,
    send_from_directory,
    stream_with_context,
)

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

# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    static_folder="frontend",
    static_url_path=""
)

# ============================================================
# DOWNLOAD HAARCASCADE
# ============================================================

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


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model_path = (
        BEST_MODEL_FINAL
        if os.path.exists(BEST_MODEL_FINAL)
        else BEST_MODEL_PHASE1
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            "Không tìm thấy model."
        )

    model = load_model(
    model_path,
    compile=False,
    safe_mode=False
)

    return model


# ============================================================
# DECODE BASE64 IMAGE
# ============================================================

def decode_image(data_url):

    if data_url.startswith("data:"):
        _, data_url = data_url.split(",", 1)

    try:

        image_data = base64.b64decode(data_url)

        image_array = np.frombuffer(
            image_data,
            np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        return image

    except Exception:

        return None


# ============================================================
# PREDICT FRAME
# ============================================================

def predict_frame(
    frame,
    model,
    face_cascade,
    use_mediapipe=False,
    face_detector=None,
):

    if frame is None:

        return {
            "status": "error",
            "message": "Ảnh không hợp lệ",
        }

    ih, iw = frame.shape[:2]

    faces = []

    # ========================================================
    # MEDIAPIPE DETECTION
    # ========================================================

    if use_mediapipe and face_detector is not None:

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_detector.process(rgb)

        if results.detections:

            for d in results.detections:

                box = (
                    d.location_data
                    .relative_bounding_box
                )

                x = int(box.xmin * iw)
                y = int(box.ymin * ih)
                w = int(box.width * iw)
                h = int(box.height * ih)

                x = max(0, x)
                y = max(0, y)

                w = min(w, iw - x)
                h = min(h, ih - y)

                if w > 30 and h > 30:
                    faces.append((x, y, w, h))

    # ========================================================
    # HAARCASCADE DETECTION
    # ========================================================

    else:

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        detected = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        if len(detected) > 0:
            faces = list(detected)

    # ========================================================
    # NO FACE
    # ========================================================

    if len(faces) == 0:

        return {
            "status": "no_face",
            "message": "Không tìm thấy khuôn mặt",
        }

    # ========================================================
    # LẤY KHUÔN MẶT ĐẦU TIÊN
    # ========================================================

    x, y, w, h = faces[0]

    face = frame[y:y+h, x:x+w]

    if face.size == 0:

        return {
            "status": "error",
            "message": "Không cắt được khuôn mặt",
        }

    # ========================================================
    # PREPROCESS
    # ========================================================

    face = cv2.resize(
        face,
        IMG_SIZE
    )

    face = img_to_array(face)

    face = face.astype("float32")

    face *= RESCALE

    face = np.expand_dims(face, axis=0)

    # ========================================================
    # PREDICT
    # ========================================================

    preds = model.predict(
        face,
        verbose=0
    )[0]

    class_id = int(np.argmax(preds))

    confidence = float(preds[class_id])

    label = CLASS_NAMES[class_id]

    # ========================================================
    # CONFIDENCE CHECK
    # ========================================================

    if confidence < DETECTION_CONFIDENCE:

        label = "Không chắc chắn"

    # ========================================================
    # COLOR
    # ========================================================

    color = (0, 255, 0)

    if label.lower() == "without_mask":
        color = (0, 0, 255)

    elif label.lower() == "incorrect_mask":
        color = (0, 255, 255)

    # ========================================================
    # DRAW
    # ========================================================

    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        color,
        2,
    )

    text = f"{label}: {confidence * 100:.2f}%"

    cv2.putText(
        frame,
        text,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    # ========================================================
    # ENCODE IMAGE
    # ========================================================

    _, buffer = cv2.imencode(
        ".jpg",
        frame
    )

    image_base64 = base64.b64encode(
        buffer
    ).decode("utf-8")

    return {
        "status": "success",
        "label": label,
        "confidence": confidence,
        "image": image_base64,
    }


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return send_from_directory(
        "frontend",
        "index.html"
    )


@app.route("/about.html")
def about():
    return send_from_directory(
        "frontend",
        "about.html"
    )


# ============================================================
# VIDEO STREAM
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(

        stream_with_context(

            realtime_detect.generate_frames(

                app.config["MODEL"],

                app.config["FACE_CASCADE"],

                use_mediapipe=app.config.get(
                    "USE_MEDIAPIPE",
                    False
                ),

                face_detector=app.config.get(
                    "FACE_DETECTOR",
                    None
                ),
            )
        ),

        mimetype=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        ),
    )


# ============================================================
# API PREDICT
# ============================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():

    payload = request.get_json(
        force=True
    )

    if (
        not payload
        or "image" not in payload
    ):

        return jsonify({

            "status": "error",

            "message": "Thiếu image"

        }), 400

    frame = decode_image(
        payload["image"]
    )

    if frame is None:

        return jsonify({

            "status": "error",

            "message": "Không decode được ảnh"

        }), 400

    result = predict_frame(

        frame,

        app.config["MODEL"],

        app.config["FACE_CASCADE"],

        use_mediapipe=app.config.get(
            "USE_MEDIAPIPE",
            False
        ),

        face_detector=app.config.get(
            "FACE_DETECTOR",
            None
        ),
    )

    return jsonify(result)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n🚀 Khởi tạo realtime detection...")

    (
        model,
        face_cascade,
        use_mediapipe,
        face_detector,
    ) = realtime_detect.init_realtime_resources()

    app.config["MODEL"] = model

    app.config["FACE_CASCADE"] = face_cascade

    app.config["USE_MEDIAPIPE"] = use_mediapipe

    app.config["FACE_DETECTOR"] = face_detector

    print("\n✅ Flask server running...")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )