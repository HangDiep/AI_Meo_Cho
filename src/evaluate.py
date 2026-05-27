"""
evaluate.py — Đánh giá mô hình toàn diện trên tập test.

Tạo ra:
1. Confusion Matrix
2. Classification Report
3. Ảnh dự đoán đúng / sai
4. Báo cáo tổng hợp (.txt)

Cách chạy:
    cd d:/AI/Khuau_trang
    python src/evaluate.py
"""

import os
import sys
import io

# =========================================================
# FIX UTF-8 CHO WINDOWS
# =========================================================
if sys.platform == "win32":

    if hasattr(sys.stdout, "reconfigure"):

        sys.stdout.reconfigure(
            encoding="utf-8",
            errors="replace"
        )

        sys.stderr.reconfigure(
            encoding="utf-8",
            errors="replace"
        )

    else:

        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace"
        )

        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer,
            encoding="utf-8",
            errors="replace"
        )

# =========================================================
# ADD PROJECT ROOT TO PATH
# =========================================================
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

# =========================================================
# IMPORTS
# =========================================================
import numpy as np
import keras

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)

from src.config import (
    BEST_MODEL_FINAL,
    PLOTS_DIR,
    REPORTS_DIR,
    CLASS_NAMES,
)

from src.data_preprocessing import (
    create_data_generators
)

from src.utils import (
    plot_confusion_matrix,
    plot_sample_predictions,
    save_report,
)

# =========================================================
# EVALUATE
# =========================================================
def evaluate():

    # =====================================================
    # CHECK MODEL
    # =====================================================
    model_path = BEST_MODEL_FINAL

    if not os.path.exists(model_path):

        print("\n❌ Không tìm thấy model:")
        print(model_path)

        return

    # =====================================================
    # LOAD MODEL
    # =====================================================
    print("\n📦 Đang load model:")
    print(model_path)

    model = keras.models.load_model(
        model_path,
        compile=False
    )

    print("\n✅ Load model thành công!")

    # =====================================================
    # COMPILE
    # =====================================================
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # =====================================================
    # DATA GENERATOR
    # =====================================================
    print("\n📂 Đang tạo test generator...")

    _, _, test_gen = create_data_generators()

    # =====================================================
    # PREDICT
    # =====================================================
    print("\n🔮 Đang dự đoán...")

    y_pred_probs = model.predict(
        test_gen,
        verbose=1
    )

    y_pred = np.argmax(
        y_pred_probs,
        axis=1
    )

    y_true = test_gen.classes

    # =====================================================
    # EVALUATE
    # =====================================================
    print("\n📊 Đánh giá mô hình...")

    loss, accuracy = model.evaluate(
        test_gen,
        verbose=0
    )

    print(f"\n{'=' * 60}")
    print("TEST RESULTS")
    print(f"{'=' * 60}")

    print(f"Loss     : {loss:.4f}")

    print(
        f"Accuracy : "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(f"{'=' * 60}")

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================
    print("\n📌 Tạo confusion matrix...")

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plot_confusion_matrix(
        cm,
        CLASS_NAMES,
        PLOTS_DIR
    )

    print("\nConfusion Matrix:")
    print(cm)

    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================
    print("\n📌 Classification Report:")

    report_str = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )

    print(report_str)

    # =====================================================
    # SAMPLE PREDICTIONS
    # =====================================================
    print("\n🖼️ Tạo ảnh dự đoán mẫu...")

    test_gen.reset()

    images_batch, labels_batch = next(test_gen)

    batch_preds_probs = model.predict(
        images_batch,
        verbose=0
    )

    batch_preds = np.argmax(
        batch_preds_probs,
        axis=1
    )

    labels_batch = np.argmax(
        labels_batch,
        axis=1
    )

    # =====================================================
    # CORRECT
    # =====================================================
    correct_mask = (
        batch_preds ==
        labels_batch.astype(int)
    )

    if correct_mask.sum() > 0:

        correct_idx = np.where(
            correct_mask
        )[0][:16]

        plot_sample_predictions(
            images_batch[correct_idx],
            labels_batch[correct_idx],
            batch_preds[correct_idx],
            np.max(
                batch_preds_probs[correct_idx],
                axis=1
            ),
            CLASS_NAMES,
            PLOTS_DIR,
            filename="correct_predictions.png"
        )

    # =====================================================
    # WRONG
    # =====================================================
    wrong_mask = ~correct_mask

    if wrong_mask.sum() > 0:

        wrong_idx = np.where(
            wrong_mask
        )[0][:16]

        plot_sample_predictions(
            images_batch[wrong_idx],
            labels_batch[wrong_idx],
            batch_preds[wrong_idx],
            np.max(
                batch_preds_probs[wrong_idx],
                axis=1
            ),
            CLASS_NAMES,
            PLOTS_DIR,
            filename="wrong_predictions.png"
        )

    else:

        print("\n🎉 Không có dự đoán sai!")

    # =====================================================
    # SAVE REPORT
    # =====================================================
    print("\n💾 Đang lưu báo cáo...")

    full_report = []

    full_report.append("=" * 60)
    full_report.append("BÁO CÁO ĐÁNH GIÁ MÔ HÌNH")
    full_report.append("=" * 60)

    full_report.append(f"\nModel:")
    full_report.append(model_path)

    full_report.append(f"\nLoss:")
    full_report.append(f"{loss:.4f}")

    full_report.append(f"\nAccuracy:")
    full_report.append(
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    full_report.append("\n")
    full_report.append(report_str)

    save_report(
        "\n".join(full_report),
        REPORTS_DIR,
        "evaluation_report.txt"
    )

    # =====================================================
    # DONE
    # =====================================================
    print("\n✅ ĐÁNH GIÁ HOÀN TẤT!")

    print(
        f"\n📄 Báo cáo:"
        f"\n{os.path.join(REPORTS_DIR, 'evaluation_report.txt')}"
    )

    print(
        f"\n📊 Biểu đồ:"
        f"\n{PLOTS_DIR}"
    )


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    try:

        evaluate()

    except Exception as e:

        print("\n❌ LỖI:")
        print(e)