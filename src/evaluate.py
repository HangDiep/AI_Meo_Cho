"""
evaluate.py — Đánh giá mô hình toàn diện trên tập test.

Tạo ra:
1. Confusion Matrix (heatmap)
2. Classification Report (Precision, Recall, F1)
3. ROC Curve + AUC
4. Ảnh dự đoán đúng / sai
5. Báo cáo tổng hợp (.txt)

Cách chạy:
    cd d:/AI/Khuau_trang
    python src/evaluate.py
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

from src.config import (
    BEST_MODEL_FINAL, BEST_MODEL_PHASE1,
    PLOTS_DIR, REPORTS_DIR, CLASS_NAMES,
)
from src.data_preprocessing import create_data_generators
from src.utils import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_sample_predictions,
    save_report,
)


def evaluate():
    """Chạy toàn bộ quy trình đánh giá."""

    # =================================================================
    # LOAD MODEL
    # =================================================================
    model_path = BEST_MODEL_FINAL
    if not os.path.exists(model_path):
        model_path = BEST_MODEL_PHASE1
    if not os.path.exists(model_path):
        print("❌ Không tìm thấy model! Hãy chạy train.py trước.")
        return

    print(f"\n📦 Loading model: {model_path}")
    model = tf.keras.models.load_model(
    model_path,
    compile=False
)

    # =================================================================
    # TẠO TEST GENERATOR
    # =================================================================
    _, _, test_gen = create_data_generators()

    # =================================================================
    # DỰ ĐOÁN
    # =================================================================
    print("\n🔮 Dự đoán trên tập test...")

    # Lấy predictions
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred_probs = y_pred_probs.flatten()
    y_pred = (y_pred_probs >= 0.5).astype(int)

    # Lấy true labels
    y_true = test_gen.classes

    # =================================================================
    # 1. TEST ACCURACY & LOSS
    # =================================================================
    loss, accuracy = model.evaluate(test_gen, verbose=0)

    print(f"\n{'=' * 50}")
    print(f"  TEST RESULTS")
    print(f"{'=' * 50}")
    print(f"  Loss:     {loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"{'=' * 50}")

    # =================================================================
    # 2. CONFUSION MATRIX
    # =================================================================
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, CLASS_NAMES, PLOTS_DIR)

    print(f"\nConfusion Matrix:")
    print(f"  {CLASS_NAMES[0]:>15}  {CLASS_NAMES[1]:>15}  ← Predicted")
    print(f"  {cm[0][0]:>15}  {cm[0][1]:>15}  | {CLASS_NAMES[0]} (actual)")
    print(f"  {cm[1][0]:>15}  {cm[1][1]:>15}  | {CLASS_NAMES[1]} (actual)")

    # =================================================================
    # 3. CLASSIFICATION REPORT
    # =================================================================
    report_str = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    )
    print(f"\nClassification Report:")
    print(report_str)

    # =================================================================
    # 4. ROC CURVE & AUC
    # =================================================================
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    auc_score = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, auc_score, PLOTS_DIR)
    print(f"  AUC Score: {auc_score:.4f}")

    # =================================================================
    # 5. ẢNH DỰ ĐOÁN ĐÚNG / SAI
    # =================================================================
    print("\n🖼️  Tạo ảnh mẫu dự đoán...")

    # Lấy batch ảnh thật từ test generator
    test_gen.reset()
    images_batch, labels_batch = next(test_gen)

    batch_preds_probs = model.predict(images_batch, verbose=0).flatten()
    batch_preds = (batch_preds_probs >= 0.5).astype(int)

    # Ảnh dự đoán đúng
    correct_mask = (batch_preds == labels_batch.astype(int))
    if correct_mask.sum() > 0:
        correct_idx = np.where(correct_mask)[0][:16]
        plot_sample_predictions(
            images_batch[correct_idx],
            labels_batch[correct_idx],
            batch_preds[correct_idx],
            batch_preds_probs[correct_idx],
            CLASS_NAMES, PLOTS_DIR,
            filename="correct_predictions.png",
        )

    # Ảnh dự đoán sai
    wrong_mask = ~correct_mask
    if wrong_mask.sum() > 0:
        wrong_idx = np.where(wrong_mask)[0][:16]
        plot_sample_predictions(
            images_batch[wrong_idx],
            labels_batch[wrong_idx],
            batch_preds[wrong_idx],
            batch_preds_probs[wrong_idx],
            CLASS_NAMES, PLOTS_DIR,
            filename="wrong_predictions.png",
        )
    else:
        print("  Không có dự đoán sai trong batch này! 🎉")

    # =================================================================
    # 6. BÁO CÁO TỔNG HỢP
    # =================================================================
    full_report = []
    full_report.append("=" * 60)
    full_report.append("  BÁO CÁO ĐÁNH GIÁ MÔ HÌNH MASK DETECTION")
    full_report.append("=" * 60)
    full_report.append(f"\nModel: {model_path}")
    full_report.append(f"\nTest Loss:     {loss:.4f}")
    full_report.append(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    full_report.append(f"AUC Score:     {auc_score:.4f}")
    full_report.append(f"\n{'─' * 40}")
    full_report.append("Confusion Matrix:")
    full_report.append(f"  TP (With_mask đúng):    {cm[0][0]}")
    full_report.append(f"  FN (With_mask sai):     {cm[0][1]}")
    full_report.append(f"  FP (Without_mask sai):  {cm[1][0]}")
    full_report.append(f"  TN (Without_mask đúng): {cm[1][1]}")
    full_report.append(f"\n{'─' * 40}")
    full_report.append("Classification Report:")
    full_report.append(report_str)
    full_report.append(f"\n{'─' * 40}")
    full_report.append("Biểu đồ đã lưu:")
    full_report.append(f"  - {os.path.join(PLOTS_DIR, 'confusion_matrix.png')}")
    full_report.append(f"  - {os.path.join(PLOTS_DIR, 'roc_curve.png')}")
    full_report.append(f"  - {os.path.join(PLOTS_DIR, 'training_history.png')}")
    full_report.append(f"  - {os.path.join(PLOTS_DIR, 'correct_predictions.png')}")
    full_report.append(f"  - {os.path.join(PLOTS_DIR, 'wrong_predictions.png')}")

    save_report("\n".join(full_report), REPORTS_DIR, "evaluation_report.txt")

    print(f"\n✅ ĐÁNH GIÁ HOÀN TẤT!")
    print(f"   Báo cáo: {os.path.join(REPORTS_DIR, 'evaluation_report.txt')}")
    print(f"   Biểu đồ: {PLOTS_DIR}")


if __name__ == "__main__":
    evaluate()
