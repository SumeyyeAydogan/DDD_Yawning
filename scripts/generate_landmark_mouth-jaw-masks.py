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
from typing import Iterable, List

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError

import mediapipe as mp


# ------------ CONFIG ------------
PROJECT_ROOT = Path(__file__).parent.parent

SOURCE_SPLIT_ROOT = PROJECT_ROOT / "ydd_splitted_dataset"
TARGET_SPLIT_ROOT = PROJECT_ROOT / "ydd_splitted_dataset_landmark_mouth-jaw-11"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
CLASSES = ("NoYawn", "Yawn")
SPLITS = ("train", "val", "test")

# Background value for non-ROI pixels: 0 = black
BACKGROUND_VALUE = 0.0  # you can set 0.2, 0.5, etc. if you want gray background

# Padding (in pixels) around the mouth+jaw rectangular ROI
ROI_PADDING_PX = 11


mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)

# Example landmark index sets for eyes + mouth (MediaPipe FaceMesh)
# You can tweak these if you want larger/smaller regions.
LEFT_EYE_IDX: List[int] = [33, 7, 163, 144, 145, 153, 154, 155, 133]
RIGHT_EYE_IDX: List[int] = [263, 249, 390, 373, 374, 380, 381, 382, 362]
MOUTH_IDX: List[int] = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]

# Jaw / chin landmarks (same as in `model_comparision_integral.py`)
JAW_IDX: List[int] = [
    152,                 # chin center
    377, 400, 378, 379,  # right jaw
    148, 176, 149, 150,  # left jaw
]

# Combined ROI landmarks (mouth + jaw)
ROI_IDX: List[int] = MOUTH_IDX + JAW_IDX


def create_mouth_jaw_mask(image_np: np.ndarray, face_landmarks) -> np.ndarray:
    """
    image_np: (H, W, 3) RGB uint8
    face_landmarks: mp_face_mesh result for a single face

    Adapted from `create_landmark_mask` in `model_comparision_integral.py`.
    Builds a single rectangular ROI covering **mouth + jaw** landmarks
    with some padding.

    Returns:
        (H, W) float32 mask in {0, 1} where:
            1 = mouth + jaw rectangular ROI
            0 = everything else
    """
    h, w = image_np.shape[:2]

    xs: List[int] = []
    ys: List[int] = []
    for idx in ROI_IDX:
        lm = face_landmarks.landmark[idx]

        # Clamp normalized coords (mediapipe bazen çok az taşabiliyor)
        lx = max(0.0, min(1.0, lm.x))
        ly = max(0.0, min(1.0, lm.y))

        x = int(round(lx * (w - 1)))
        y = int(round(ly * (h - 1)))

        xs.append(x)
        ys.append(y)

    if not xs:
        # Fallback: empty mask
        return np.zeros((h, w), dtype=np.float32)

    pad = int(ROI_PADDING_PX)

    x0 = max(0, min(xs) - pad)
    y0 = max(0, min(ys) - pad)

    # end‑exclusive bounds for slicing
    x1 = min(w, max(xs) + pad + 1)
    y1 = min(h, max(ys) + pad + 1)

    # ensure at least 1px ROI
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)

    mask = np.zeros((h, w), dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    return mask


def apply_mask_to_image(img: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    Apply a binary mask (0/1) to an RGB PIL image.

    Eye+mouth pixels remain; everything else is set to BACKGROUND_VALUE.
    """
    img_np = np.array(img).astype(np.float32) / 255.0  # (H, W, 3)
    if mask.ndim == 2:
        mask_3 = np.stack([mask] * 3, axis=-1)
    else:
        mask_3 = mask

    bg = float(BACKGROUND_VALUE)
    masked = img_np * mask_3 + (1.0 - mask_3) * bg
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
    img_np = np.array(img)  # RGB uint8

    # MediaPipe expects RGB uint8 array
    results = mp_face_mesh.process(img_np)

    if not results.multi_face_landmarks:
        # If no face is found, simply copy the original image.
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path)
        print(f"[WARN] No face detected, copied original: {src_path}")
        return

    face = results.multi_face_landmarks[0]
    mask = create_mouth_jaw_mask(img_np, face)  # (H, W) in {0, 1}

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


