import gradio as gr
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# ====================== CONFIG ======================
MODEL_PATH = "models/best_final.h5"
HAARCASCADE_PATH = "haarcascade/haarcascade_frontalface_default.xml"

# Load model
if not os.path.exists(MODEL_PATH):
    print(f"❌ Không tìm thấy model tại: {MODEL_PATH}")
    exit(1)

model = load_model(MODEL_PATH)
print(f"✅ Đã load model thành công: {MODEL_PATH}")

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
if face_cascade.empty():
    print(f"❌ Không load được Haar Cascade: {HAARCASCADE_PATH}")
    exit(1)
print("✅ Đã load Haar Cascade")

def preprocess_face(face_img, target_size=(224, 224)):
    if face_img is None or face_img.size == 0:
        return None
    face = cv2.resize(face_img, target_size)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    return face


def detect_mask_in_image(image):
    if image is None:
        return None, "❌ Không nhận được ảnh!"

    output_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(60, 60)
    )

    results = []

    for (x, y, w, h) in faces:
        face_img = image[y:y+h, x:x+w]
        face_input = preprocess_face(face_img)

        if face_input is None:
            continue

        prediction = model.predict(face_input, verbose=0)[0][0]
        
        if prediction > 0.5:
            label = "Không khẩu trang"
            color = (0, 0, 255)
            confidence = prediction * 100
        else:
            label = "Có khẩu trang"
            color = (0, 255, 0)
            confidence = (1 - prediction) * 100

        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 3)
        cv2.putText(output_image, f"{label} ({confidence:.1f}%)", 
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        results.append(label)

    total = len(faces)
    with_mask = sum(1 for r in results if "Có" in r)
    summary = f"📊 Tổng: {total} người | ✅ Có khẩu trang: {with_mask} | ❌ Không: {total - with_mask}"

    return output_image, summary


# ====================== GRADIO INTERFACE (Gradio 6.x) ======================
with gr.Blocks(title="Nhận Diện Khẩu Trang") as interface:
    gr.Markdown("# 🎭 Nhận Diện Khẩu Trang - Upload Ảnh")
    gr.Markdown("Tải ảnh lên để hệ thống tự động phát hiện khuôn mặt và nhận diện khẩu trang.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="📤 Tải ảnh lên")
            btn = gr.Button("🔍 Nhận diện", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(type="numpy", label="📸 Kết quả nhận diện")
            output_text = gr.Textbox(label="📋 Thống kê")

    btn.click(
        fn=detect_mask_in_image,
        inputs=input_image,
        outputs=[output_image, output_text]
    )

if __name__ == "__main__":
    print("🚀 Đang khởi động giao diện Upload Ảnh...")
    print("Truy cập tại: http://127.0.0.1:7860")
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )