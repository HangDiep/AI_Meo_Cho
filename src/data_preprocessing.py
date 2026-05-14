"""
data_preprocessing.py — Phân tích dataset (EDA) và xây dựng Data Pipeline.

Chạy file này để:
1. Thống kê dataset
2. Vẽ biểu đồ phân bố
3. Hiển thị ảnh mẫu + ảnh sau augmentation
4. Tạo data generators cho train/val/test

Cách chạy:
    cd d:/AI/Khuau_trang
    python src/data_preprocessing.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Thêm thư mục gốc vào path để import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMG_SIZE, BATCH_SIZE, RESCALE, AUGMENTATION,
    PLOTS_DIR, REPORTS_DIR, CLASS_NAMES,
)
from src.utils import count_images, save_report


# =============================================================================
# 1. THỐNG KÊ DATASET
# =============================================================================

def analyze_dataset():
    """Thống kê số lượng ảnh trong mỗi tập và mỗi class."""

    print("=" * 60)
    print("        THỐNG KÊ DATASET")
    print("=" * 60)

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("        THỐNG KÊ DATASET")
    report_lines.append("=" * 60)

    total_all = 0
    for split_name, split_dir in [("Train", TRAIN_DIR), ("Val", VAL_DIR), ("Test", TEST_DIR)]:
        counts = count_images(split_dir)
        total = sum(counts.values())
        total_all += total

        line = f"\n[{split_name}] Tổng: {total}"
        print(line)
        report_lines.append(line)

        for cls, num in counts.items():
            pct = (num / total * 100) if total > 0 else 0
            detail = f"  - {cls}: {num} ảnh ({pct:.1f}%)"
            print(detail)
            report_lines.append(detail)

    summary = f"\nTổng toàn bộ dataset: {total_all} ảnh"
    print(summary)
    report_lines.append(summary)
    print("=" * 60)

    # Lưu báo cáo
    save_report("\n".join(report_lines), REPORTS_DIR, "eda_report.txt")

    return total_all


# =============================================================================
# 2. VẼ BIỂU ĐỒ PHÂN BỐ
# =============================================================================

def plot_distribution():
    """Vẽ bar chart phân bố ảnh theo class và theo tập."""

    splits = {"Train": TRAIN_DIR, "Val": VAL_DIR, "Test": TEST_DIR}
    data = {}
    for name, path in splits.items():
        data[name] = count_images(path)

    classes = CLASS_NAMES
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (split_name, counts) in enumerate(data.items()):
        values = [counts.get(cls, 0) for cls in classes]
        bars = ax.bar(x + i * width, values, width, label=split_name)
        # Hiển thị số trên mỗi cột
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Class", fontsize=13)
    ax.set_ylabel("Số lượng ảnh", fontsize=13)
    ax.set_title("Phân bố ảnh trong Dataset", fontsize=16, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "eda_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Đã lưu biểu đồ phân bố: {save_path}")


# =============================================================================
# 3. HIỂN THỊ ẢNH MẪU
# =============================================================================

def show_sample_images(num_per_class=4):
    """Hiển thị grid ảnh mẫu từ tập train."""

    fig, axes = plt.subplots(2, num_per_class, figsize=(4 * num_per_class, 8))

    for row, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(TRAIN_DIR, class_name)
        images = sorted(os.listdir(class_dir))[:num_per_class]

        for col, img_name in enumerate(images):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                axes[row][col].imshow(img)
                axes[row][col].set_title(f"{class_name}", fontsize=11, fontweight="bold")
            except Exception as e:
                axes[row][col].text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center")
            axes[row][col].axis("off")

    plt.suptitle("Ảnh mẫu từ tập Train", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "eda_sample_images.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Đã lưu ảnh mẫu: {save_path}")


# =============================================================================
# 4. KIỂM TRA KÍCH THƯỚC ẢNH
# =============================================================================

def analyze_image_sizes(max_check=200):
    """Phân tích kích thước ảnh trong dataset."""

    widths, heights = [], []

    class_dir = os.path.join(TRAIN_DIR, CLASS_NAMES[0])
    images = os.listdir(class_dir)[:max_check]

    for img_name in images:
        try:
            img = Image.open(os.path.join(class_dir, img_name))
            w, h = img.size
            widths.append(w)
            heights.append(h)
        except Exception:
            continue

    if widths:
        print(f"\n[PHÂN TÍCH KÍCH THƯỚC] (kiểm tra {len(widths)} ảnh)")
        print(f"  Width  — Min: {min(widths)}, Max: {max(widths)}, Avg: {np.mean(widths):.0f}")
        print(f"  Height — Min: {min(heights)}, Max: {max(heights)}, Avg: {np.mean(heights):.0f}")


# =============================================================================
# 5. TẠO DATA GENERATORS
# =============================================================================

def create_data_generators():
    """
    Tạo data generators cho train, val, test.
    
    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Train: có augmentation
    train_datagen = ImageDataGenerator(
        rescale=RESCALE,
        **AUGMENTATION,
    )

    # Val & Test: chỉ rescale
    val_datagen = ImageDataGenerator(rescale=RESCALE)
    test_datagen = ImageDataGenerator(rescale=RESCALE)

    print("\n[DATA GENERATORS]")

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    # In label mapping
    print(f"\nLabel mapping: {train_generator.class_indices}")

    return train_generator, val_generator, test_generator


# =============================================================================
# 6. TRỰC QUAN AUGMENTATION
# =============================================================================

def visualize_augmentation(num_samples=8):
    """Hiển thị ảnh gốc vs ảnh sau augmentation."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

    # Lấy 1 ảnh mẫu
    sample_class = CLASS_NAMES[0]
    class_dir = os.path.join(TRAIN_DIR, sample_class)
    sample_img_name = os.listdir(class_dir)[0]
    sample_img_path = os.path.join(class_dir, sample_img_name)

    img = load_img(sample_img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = img_array.reshape((1,) + img_array.shape)

    # Tạo augmented versions
    datagen = ImageDataGenerator(**AUGMENTATION)

    fig, axes = plt.subplots(2, num_samples // 2, figsize=(4 * (num_samples // 2), 8))
    axes = axes.flatten()

    # Ảnh gốc
    axes[0].imshow(img)
    axes[0].set_title("GỐC", fontsize=12, fontweight="bold", color="blue")
    axes[0].axis("off")

    # Augmented
    it = datagen.flow(img_array, batch_size=1)
    for i in range(1, num_samples):
        aug_img = next(it)[0].astype("uint8")
        axes[i].imshow(aug_img)
        axes[i].set_title(f"Augmented #{i}", fontsize=10)
        axes[i].axis("off")

    plt.suptitle("Ảnh gốc vs Augmented", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "eda_augmentation.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Đã lưu trực quan augmentation: {save_path}")


# =============================================================================
# MAIN — Chạy toàn bộ EDA
# =============================================================================

if __name__ == "__main__":
    print("\n🔍 BẮT ĐẦU PHÂN TÍCH DATASET...\n")

    # 1. Thống kê
    analyze_dataset()

    # 2. Phân bố
    plot_distribution()

    # 3. Ảnh mẫu
    show_sample_images()

    # 4. Kích thước ảnh
    analyze_image_sizes()

    # 5. Trực quan augmentation
    visualize_augmentation()

    # 6. Test data generators
    train_gen, val_gen, test_gen = create_data_generators()

    print("\n✅ HOÀN TẤT PHÂN TÍCH DATASET!")
    print(f"   Xem kết quả tại: {PLOTS_DIR}")
    print(f"   Xem báo cáo tại: {REPORTS_DIR}")
