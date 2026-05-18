"""
Generate eye+mouth–focused versions of images using facial landmarks
without OpenCV.

Uses:
- MediaPipe FaceMesh for facial landmarks
- PIL (Pillow) + NumPy for image IO and polygon filling

Input  structure (existing splits):
    splitted_dataset/
        train/NotDrowsy/*.jpg|png
        train/Drowsy/*.jpg|png
        val/NotDrowsy/*.jpg|png
        val/Drowsy/*.jpg|png
        test/...

Output structure:
    splitted_dataset_landmark/
        (same subfolders and filenames as above, but with pixels
         outside eye+mouth regions darkened to 0)

You can then point your training pipeline to `splitted_dataset_landmark`
instead of `splitted_dataset`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, UnidentifiedImageError

import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.mask_helpers import (
    create_landmark_mask,
    image_to_float01_rgb,
    image_to_uint8_rgb,
    img_size_from_rgb,
)


# ------------ CONFIG ------------
PROJECT_ROOT = Path(__file__).parent.parent

SOURCE_SPLIT_ROOT = PROJECT_ROOT / "ydd_splitted_dataset"
TARGET_SPLIT_ROOT = PROJECT_ROOT / "ydd_splitted_dataset_roi"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
CLASSES = ("NoYawn", "Yawn")
SPLITS = ("train", "val", "test")

# Shared mask params (keys match auto_optimize / statistics scripts)
MASK_CONFIG = {
    "roi_padding_px": 6,
    "roi_keep_aspect_pad_x_min_scale": 0.2,
    "background_mask_value": 0.0,  # 0 = black outside ROI; 0.2 = soft mask
}

def apply_mask_to_image(img: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    Apply a binary mask (0/1) to an RGB PIL image.

    Eye+mouth pixels remain; everything else is set to BACKGROUND_VALUE.
    """
    img_float01 = image_to_float01_rgb(np.array(img))
    if mask.ndim == 2:
        mask_3 = np.stack([mask] * 3, axis=-1)
    else:
        mask_3 = mask

    bg = float(MASK_CONFIG["background_mask_value"])
    masked = img_float01 * mask_3 + (1.0 - mask_3) * bg
    out_np = (np.clip(masked, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(out_np)


def process_one_image(src_path: Path, dst_path: Path) -> None:
    """Detect landmarks, build eye+mouth mask and write masked image."""
    try:
        img = Image.open(src_path).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        # Corrupted or non-image file → just skip or copy original if you want
        print(f"[WARN] Skipping unreadable image: {src_path} ({e})")
        return  # veya: dst_path.parent.mkdir(...); shutil.copy(src_path, dst_path); return
    image_rgb_uint8 = image_to_uint8_rgb(np.array(img))
    img_size = img_size_from_rgb(image_rgb_uint8)

    mask = create_landmark_mask(image_rgb_uint8, img_size, MASK_CONFIG)
    if mask is None:
        # If no face is found, simply copy the original image.
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path)
        print(f"[WARN] No face detected, copied original: {src_path}")
        return

    masked_img = apply_mask_to_image(img, mask)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    masked_img.save(dst_path)
    print(f"[OK] Processed: {src_path} -> {dst_path}")


def process_split(split: str) -> None:
    """Process one split: 'train', 'val', or 'test'."""
    for cls_name in CLASSES:
        src_dir = SOURCE_SPLIT_ROOT / split / cls_name
        if not src_dir.exists():
            print(f"[WARN] Missing folder for split='{split}', class='{cls_name}': {src_dir}")
            continue

        for img_path in src_dir.rglob("*"):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue

            rel = img_path.relative_to(SOURCE_SPLIT_ROOT)
            dst_path = TARGET_SPLIT_ROOT / rel
            process_one_image(img_path, dst_path)


def main():
    print(f"[INFO] Source root: {SOURCE_SPLIT_ROOT}")
    print(f"[INFO] Target root: {TARGET_SPLIT_ROOT}")

    for split in SPLITS:
        print(f"\n[INFO] Processing split: {split}")
        process_split(split)

    print("\n[DONE] Landmark‑based masked dataset created at:")
    print(f"       {TARGET_SPLIT_ROOT}")


if __name__ == "__main__":
    main()