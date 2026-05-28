import base64
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

import cv2
import numpy as np
import tensorflow as tf

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

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

    if not os.path.exists(BEST_MODEL_FINAL):
        raise FileNotFoundError(
            f"Không tìm thấy model tại {BEST_MODEL_FINAL}."
        )

    model = tf.keras.models.load_model(
        BEST_MODEL_FINAL,
        compile=False
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

    except Exception as e:
        print(f"❌ Lỗi API Upload: {e}")
        return jsonify({
            "status": "error",
            "message": "Lỗi server khi xử lý ảnh"
        }), 500


# ============================================================
# PREDICT FRAME
# ============================================================

def predict_frame(
    frame,
    model,
    face_cascade,
    use_mediapipe=True,
    face_detector=None,
):

    try:
        if frame is None:
            return {"status": "error", "message": "Ảnh không hợp lệ"}

        ih, iw = frame.shape[:2]
        faces = []

        # ================== 1. MEDIAPIPE (Ưu tiên) ==================
        if use_mediapipe and face_detector is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            detection_result = face_detector.detect(mp_image)

            if detection_result.detections:
                for detection in detection_result.detections:
                    bbox = detection.bounding_box
                    x = max(0, int(bbox.origin_x))
                    y = max(0, int(bbox.origin_y))
                    w = int(bbox.width)
                    h = int(bbox.height)

                    if w > 45 and h > 45:
                        faces.append((x, y, w, h))

        # ================== 2. HAAR CASCADE (Fallback) ==================
        if len(faces) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.02,
                minNeighbors=4,
                minSize=(40, 40)
            )
            if len(detected) > 0:
                faces = list(detected)

        # ================== KIỂM TRA KẾT QUẢ ==================
        if len(faces) == 0:
            return {"status": "no_face", "message": "Không tìm thấy khuôn mặt nào"}

        results = []

        # ================== XỬ LÝ TẤT CẢ KHUÔN MẶT ==================
        for (x, y, w, h) in faces:
            pad = 32
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(iw, x + w + pad)
            y2 = min(ih, y + h + pad)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # Preprocess
            face = cv2.resize(face_crop, IMG_SIZE)
            face = img_to_array(face)
            face = face.astype("float32") * RESCALE
            face = np.expand_dims(face, axis=0)

            # Predict
            preds = model.predict(face, verbose=0)[0]
            class_id = int(np.argmax(preds))
            confidence = float(preds[class_id])

            label = CLASS_NAMES[class_id]

            if confidence < 0.62:
                label = "Không chắc chắn"

            # Vẽ khung
            if label == "With_mask":
                color = (0, 255, 0)
            elif label == "incorrect_mask":
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 4)
            text = f"{label} {confidence*100:.1f}%"
            cv2.putText(frame, text, (x, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.82, color, 2)

            results.append({
                "label": label,
                "confidence": round(confidence * 100, 2),
                "box": [int(x), int(y), int(w), int(h)]
            })

        # ================== TRẢ VỀ KẾT QUẢ ==================
        _, buffer = cv2.imencode(".jpg", frame)
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "status": "success",
            "total_faces": len(results),        # Đếm đúng số results
            "results": results,
            "image": image_base64
        }

    except Exception as e:
        print(f"❌ Lỗi predict_frame: {e}")
        return {"status": "error", "message": f"Lỗi xử lý: {str(e)}"}




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