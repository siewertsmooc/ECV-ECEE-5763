#!/usr/bin/env python3
"""
Train YOLO on MNIST digits by converting MNIST classification samples
into a simple object-detection dataset.

Requirements:
    pip install ultralytics torchvision torch pillow numpy

Usage:
    python3 mnist_yolo_train.py

Outputs:
    mnist_yolo_dataset/
    runs/mnist_yolo/
"""

import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import datasets


# -------------------------
# Config
# -------------------------
ROOT = Path(".")
DATASET_DIR = ROOT / "mnist_yolo_dataset"
MNIST_RAW_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs" / "mnist_yolo"

TRAIN_LIMIT = 20000   # reduce for faster demo; set to 60000 for full MNIST train
VAL_LIMIT = 5000      # set to 10000 for full test set
PAD = 2               # bbox padding around visible digit pixels
IMG_EXT = ".png"

CLASS_NAMES = [str(i) for i in range(10)]


# -------------------------
# Helpers
# -------------------------
def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def make_dirs():
    for split in ["train", "val"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def find_digit_bbox(img_array: np.ndarray, pad: int = 2):
    """
    Compute a bounding box around non-zero pixels in a 28x28 MNIST image.
    Returns (x_min, y_min, x_max, y_max), inclusive-exclusive style.
    """
    ys, xs = np.where(img_array > 0)

    if len(xs) == 0 or len(ys) == 0:
        # fallback: full image
        return 0, 0, img_array.shape[1], img_array.shape[0]

    x_min = max(0, xs.min() - pad)
    y_min = max(0, ys.min() - pad)
    x_max = min(img_array.shape[1], xs.max() + 1 + pad)
    y_max = min(img_array.shape[0], ys.max() + 1 + pad)

    return x_min, y_min, x_max, y_max


def bbox_to_yolo(x_min, y_min, x_max, y_max, width, height):
    """
    Convert pixel bbox to YOLO normalized format:
    x_center y_center box_width box_height
    """
    bw = x_max - x_min
    bh = y_max - y_min
    xc = x_min + bw / 2.0
    yc = y_min + bh / 2.0

    return xc / width, yc / height, bw / width, bh / height


def save_sample(img, label, out_img_path: Path, out_lbl_path: Path, pad: int = 2):
    """
    Save MNIST PIL image and matching YOLO label file.
    """
    img_array = np.array(img)
    height, width = img_array.shape

    x_min, y_min, x_max, y_max = find_digit_bbox(img_array, pad=pad)
    xc, yc, bw, bh = bbox_to_yolo(x_min, y_min, x_max, y_max, width, height)

    img.save(out_img_path)

    with open(out_lbl_path, "w") as f:
        f.write(f"{label} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def build_split(dataset, split_name: str, max_items: int):
    img_dir = DATASET_DIR / "images" / split_name
    lbl_dir = DATASET_DIR / "labels" / split_name

    n = min(len(dataset), max_items)
    for i in range(n):
        img, label = dataset[i]
        stem = f"{split_name}_{i:06d}"
        out_img = img_dir / f"{stem}{IMG_EXT}"
        out_lbl = lbl_dir / f"{stem}.txt"
        save_sample(img, label, out_img, out_lbl, pad=PAD)

    print(f"Built {split_name}: {n} samples")


def write_dataset_yaml():
    yaml_text = f"""path: {DATASET_DIR.resolve()}
train: images/train
val: images/val

names:
"""
    for idx, name in enumerate(CLASS_NAMES):
        yaml_text += f"  {idx}: {name}\n"

    yaml_path = DATASET_DIR / "mnist.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_text)

    return yaml_path


def train_yolo(data_yaml: Path):
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=10,
        imgsz=32,
        batch=128,
        project=str(RUNS_DIR.parent),
        name=RUNS_DIR.name,
        exist_ok=True,
        pretrained=True,
        verbose=True,
        plots=True,
    )

    return model, results


def validate_yolo(model, data_yaml: Path):
    metrics = model.val(
        data=str(data_yaml),
        imgsz=32,
        split="val",
        plots=True,
    )
    return metrics


def write_summary(run_dir: Path, metrics):
    summary_path = run_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("MNIST YOLO Training Summary\n")
        f.write("===========================\n\n")
        f.write("Dataset: MNIST converted to YOLO object detection\n")
        f.write("Classes: 10 digits (0-9)\n")
        f.write("One bounding box per image around visible digit pixels\n\n")

        names_and_values = [
            ("metrics.box.map", getattr(metrics.box, "map", None)),
            ("metrics.box.map50", getattr(metrics.box, "map50", None)),
            ("metrics.box.map75", getattr(metrics.box, "map75", None)),
            ("metrics.box.mp", getattr(metrics.box, "mp", None)),
            ("metrics.box.mr", getattr(metrics.box, "mr", None)),
        ]

        for name, value in names_and_values:
            if value is not None:
                f.write(f"{name}: {value}\n")

        f.write("\nExpected YOLO result files in this run directory typically include:\n")
        f.write("- results.csv\n")
        f.write("- results.png\n")
        f.write("- confusion_matrix.png\n")
        f.write("- confusion_matrix_normalized.png\n")
        f.write("- F1_curve.png\n")
        f.write("- P_curve.png\n")
        f.write("- R_curve.png\n")
        f.write("- PR_curve.png\n")
        f.write("- weights/best.pt\n")
        f.write("- weights/last.pt\n")


def main():
    print("Preparing dataset directories...")
    ensure_clean_dir(DATASET_DIR)
    make_dirs()

    print("Downloading/loading MNIST...")
    mnist_train = datasets.MNIST(root=str(MNIST_RAW_DIR), train=True, download=True)
    mnist_val = datasets.MNIST(root=str(MNIST_RAW_DIR), train=False, download=True)

    print("Converting MNIST to YOLO format...")
    build_split(mnist_train, "train", TRAIN_LIMIT)
    build_split(mnist_val, "val", VAL_LIMIT)

    data_yaml = write_dataset_yaml()
    print(f"Wrote dataset YAML: {data_yaml}")

    print("Training YOLO...")
    model, _ = train_yolo(data_yaml)

    print("Running validation...")
    metrics = validate_yolo(model, data_yaml)

    run_dir = RUNS_DIR
    write_summary(run_dir, metrics)

    print("\nDone.")
    print(f"Dataset saved to: {DATASET_DIR.resolve()}")
    print(f"Training results saved to: {run_dir.resolve()}")
    print("\nLook for these files in the run directory:")
    print("  weights/best.pt")
    print("  weights/last.pt")
    print("  results.csv")
    print("  results.png")
    print("  PR_curve.png")
    print("  P_curve.png")
    print("  R_curve.png")
    print("  F1_curve.png")
    print("  confusion_matrix.png")
    print("  confusion_matrix_normalized.png")
    print("  summary.txt")


if __name__ == "__main__":
    main()
