# src/web_upload.py
import gradio as gr
import cv2
import numpy as np
import os

from tensorflow.keras.models import load_model

# ====================== IMPORT CONFIG ======================
from src.config import (
    BEST_MODEL_FINAL,
    HAARCASCADE_PATH,
    IMG_SIZE,
    CLASS_NAMES,
    COLOR_MASK,
    COLOR_NO_MASK,
    COLOR_INCORRECT_MASK,
)

# ====================== LOAD MODEL & CASCADE ======================
if not os.path.exists(BEST_MODEL_FINAL):
    print(f"❌ Không tìm thấy model tại: {BEST_MODEL_FINAL}")
    print("Vui lòng train model trước!")
    exit(1)

model = load_model(BEST_MODEL_FINAL, compile=False)
print(f"✅ Đã load model thành công: {BEST_MODEL_FINAL}")

face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
if face_cascade.empty():
    print(f"❌ Không load được Haar Cascade: {HAARCASCADE_PATH}")
    exit(1)
print("✅ Đã load Haar Cascade")

# ====================== PREPROCESS ======================
def preprocess_face(face_img):
    """Tiền xử lý khuôn mặt"""
    if face_img is None or face_img.size == 0:
        return None
    face = cv2.resize(face_img, IMG_SIZE)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    return face


# ====================== MAIN FUNCTION ======================
def detect_mask_in_image(image):
    if image is None:
        return None, "❌ Không nhận được ảnh!"

    output_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Face Detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        maxSize=(600, 600)
    )

    print(f"🔍 Phát hiện {len(faces)} khuôn mặt")

    mask_count = incorrect_count = no_mask_count = 0

    for (x, y, w, h) in faces:
        pad = 25
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)

        face_img = image[y1:y2, x1:x2]
        face_input = preprocess_face(face_img)

        if face_input is None:
            continue

        prediction = model.predict(face_input, verbose=0)[0]
        class_id = np.argmax(prediction)
        confidence = prediction[class_id] * 100

        if confidence < 50:
            continue

        label = CLASS_NAMES[class_id]

        if class_id == 1:           # With_mask
            color = COLOR_MASK
            mask_count += 1
        elif class_id == 0:         # incorrect_mask
            color = COLOR_INCORRECT_MASK
            incorrect_count += 1
        else:                       # Without_mask
            color = COLOR_NO_MASK
            no_mask_count += 1

        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 4)
        cv2.putText(output_image, f"{label} ({confidence:.1f}%)",
                    (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 3)

    total = len(faces)
    summary = f"""📊 THỐNG KÊ
────────────────
👥 Tổng người: {total}
✅ Đeo đúng: {mask_count}
⚠️ Đeo sai: {incorrect_count}
❌ Không đeo: {no_mask_count}"""

    return output_image, summary


# ====================== GRADIO INTERFACE ======================
with gr.Blocks(title="Nhận Diện Khẩu Trang") as interface:
    gr.Markdown("# 🎭 Nhận Diện Khẩu Trang - 3 Classes")
    gr.Markdown("Sử dụng MobileNetV2 / EfficientNetB1")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="📤 Tải ảnh lên", height=600)
            btn = gr.Button("🔍 Nhận diện", variant="primary", size="large")
        
        with gr.Column():
            output_image = gr.Image(type="numpy", label="📸 Kết quả nhận diện", height=600)
            output_text = gr.Textbox(label="📋 Thống kê", lines=8)

    btn.click(
        fn=detect_mask_in_image,
        inputs=input_image,
        outputs=[output_image, output_text]
    )

if __name__ == "__main__":
    print("🚀 Khởi động giao diện Upload Ảnh...")
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )