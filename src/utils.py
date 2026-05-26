"""
utils.py — Hàm tiện ích dùng chung cho toàn bộ dự án.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import cv2
from tensorflow.keras.preprocessing.image import img_to_array


def get_timestamp():
    """Trả về timestamp dạng string để đặt tên file."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def preprocess_face(face_img, target_size=(224, 224)):
    """
    Tiền xử lý khuôn mặt để đưa vào model.
    """
    if face_img is None or face_img.size == 0:
        return None
    face = cv2.resize(face_img, target_size)
    face = face.astype("float32") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    
    return face

def count_images(directory):
    """
    Đếm số ảnh trong mỗi class (subfolder) của một thư mục.
    
    Args:
        directory: Đường dẫn tới thư mục chứa các subfolder class
        
    Returns:
        dict: {class_name: số_ảnh}
    """
    counts = {}
    if not os.path.exists(directory):
        print(f"[WARNING] Thư mục không tồn tại: {directory}")
        return counts

    for class_name in sorted(os.listdir(directory)):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
            num_images = sum(
                1 for f in os.listdir(class_path)
                if os.path.splitext(f)[1].lower() in valid_extensions
            )
            counts[class_name] = num_images

    return counts


def plot_training_history(history, save_dir):
    """
    Vẽ biểu đồ accuracy và loss qua các epoch.
    
    Args:
        history: Keras training history object hoặc dict
        save_dir: Thư mục lưu biểu đồ
    """
    # Chuyển history object sang dict nếu cần
    if hasattr(history, "history"):
        hist = history.history
    else:
        hist = history

    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Accuracy ---
    axes[0].plot(hist["accuracy"], label="Train Accuracy", linewidth=2)
    axes[0].plot(hist["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # --- Loss ---
    axes[1].plot(hist["loss"], label="Train Loss", linewidth=2)
    axes[1].plot(hist["val_loss"], label="Val Loss", linewidth=2)
    axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Đã lưu biểu đồ training history: {save_path}")


def plot_confusion_matrix(cm, class_names, save_dir):
    """
    Vẽ confusion matrix dạng heatmap.
    
    Args:
        cm: numpy array confusion matrix
        class_names: list tên các class
        save_dir: thư mục lưu
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 16},
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted", fontsize=13)
    plt.ylabel("Actual", fontsize=13)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Đã lưu confusion matrix: {save_path}")


def plot_roc_curve(fpr, tpr, auc_score, save_dir):
    """
    Vẽ ROC curve.
    
    Args:
        fpr: False Positive Rate array
        tpr: True Positive Rate array
        auc_score: AUC score
        save_dir: thư mục lưu
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=13)
    plt.ylabel("True Positive Rate", fontsize=13)
    plt.title("ROC Curve", fontsize=16, fontweight="bold")
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Đã lưu ROC curve: {save_path}")


def plot_sample_predictions(images, true_labels, pred_labels, pred_probs,
                            class_names, save_dir, filename="sample_predictions.png",
                            num_samples=16):

    os.makedirs(save_dir, exist_ok=True)

    n = min(num_samples, len(images))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    # 🔥 FIX QUAN TRỌNG: ép về 1D array an toàn
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        ax = axes[i]
        ax.imshow(images[i])
        ax.axis("off")

        true_name = class_names[int(true_labels[i])]
        pred_name = class_names[int(pred_labels[i])]
        prob = pred_probs[i]

        correct = true_labels[i] == pred_labels[i]
        color = "green" if correct else "red"
        symbol = "✓" if correct else "✗"

        ax.set_title(
            f"{symbol} True: {true_name}\nPred: {pred_name} ({prob:.1%})",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    # Ẩn ô dư
    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Đã lưu sample predictions: {save_path}")


def save_report(text, save_dir, filename="evaluation_report.txt"):
    """Lưu báo cáo text ra file."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[OK] Đã lưu báo cáo: {save_path}")
