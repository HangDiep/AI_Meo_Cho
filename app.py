import base64
import os
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

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
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

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    return model

# ============================================================
# INIT
# ============================================================

download_haarcascade()
model = load_model()
face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)


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
# UPLOAD IMAGE API (MỚI THÊM)
# ============================================================

@app.route("/api/upload", methods=["POST"])
def api_upload():
    try:
        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "Không có file ảnh"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"status": "error", "message": "File rỗng"}), 400

        image_array = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"status": "error", "message": "Không đọc được file ảnh"}), 400

        result = predict_frame(frame, model, face_cascade)
        return jsonify(result)

    except Exception as e:
        print(f"❌ Lỗi API Upload: {e}")
        return jsonify({
            "status": "error",
            "message": "Lỗi server khi xử lý ảnh"
        }), 500


# ============================================================
# PREDICT FRAME
# ============================================================

def predict_frame(frame, model, face_cascade):
    try:
        if frame is None:
            return {"status": "error", "message": "Ảnh không hợp lệ"}

        ih, iw = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(40, 40)
        )

        if len(faces) == 0:
            return {"status": "no_face", "message": "Không tìm thấy khuôn mặt nào"}

        results = []

        for (x, y, w, h) in faces:
            pad = 25
            face_crop = frame[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]

            if face_crop.size == 0:
                continue

            face = cv2.resize(face_crop, IMG_SIZE)
            face = img_to_array(face)
            face = face.astype("float32") * RESCALE
            face = np.expand_dims(face, axis=0)

            preds = model.predict(face, verbose=0)[0]
            class_id = int(np.argmax(preds))
            confidence = float(preds[class_id])

            label = CLASS_NAMES[class_id]

            results.append({
                "label": label,
                "confidence": round(confidence * 100, 2),
                "box": [int(x), int(y), int(w), int(h)]
            })

        return {
            "status": "success",
            "total_faces": len(faces),
            "results": results
        }

    except Exception as e:
        print(f"❌ Lỗi trong predict_frame: {e}")
        return {
            "status": "error",
            "message": f"Lỗi xử lý ảnh: {str(e)}"
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