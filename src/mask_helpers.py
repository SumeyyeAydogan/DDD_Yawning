from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import mediapipe as mp


MOUTH_IDX: List[int] = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
JAW_IDX: List[int] = [152, 377, 400, 378, 379, 148, 176, 149, 150]
ROI_IDX: List[int] = MOUTH_IDX + JAW_IDX

_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)


def image_to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert an HxWx3 RGB array to uint8 (0-255) for MediaPipe FaceMesh.

    Accepts float images in [0, 1] or [0, 255] and uint8 inputs unchanged.
    """
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape {arr.shape}")
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def image_to_float01_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an HxWx3 RGB array to float32 in [0, 1] for Keras / GradCAM."""
    return image_to_uint8_rgb(image).astype(np.float32) / 255.0


def img_size_from_rgb(image_rgb_uint8: np.ndarray) -> Tuple[int, int]:
    """Return (height, width) — same convention as dataloader ``img_size=(H, W)``."""
    h, w = image_rgb_uint8.shape[:2]
    return (int(h), int(w))


def create_landmark_mask(
    image_rgb_uint8: np.ndarray,
    img_size: Tuple[int, int],
    mask_config: Dict[str, float],
) -> Optional[np.ndarray]:
    """
    Build one rectangular ROI mask based on mouth+jaw landmarks (+padding).

    Args:
        image_rgb_uint8: RGB uint8 array (H, W, 3), 0-255. Use ``image_to_uint8_rgb``.
        img_size: (height, width), e.g. (224, 224) or from ``img_size_from_rgb``.
        mask_config: roi_padding_px, roi_keep_aspect_pad_x_min_scale, background_mask_value.

    Returns (H, W) float32 mask in [bg..1.0] or None if no face.
    """
    image_rgb_uint8 = image_to_uint8_rgb(image_rgb_uint8)
    h, w = img_size
    bg = float(mask_config.get("background_mask_value", 0.0))
    pad_base = int(mask_config.get("roi_padding_px", 6))
    min_x_scale = float(mask_config.get("roi_keep_aspect_pad_x_min_scale", 0.2))

    results = _face_mesh.process(image_rgb_uint8)
    if not results.multi_face_landmarks:
        return None

    face = results.multi_face_landmarks[0]
    xs, ys = [], []
    for i in ROI_IDX:
        lm = face.landmark[i]
        lx = max(0.0, min(1.0, lm.x))
        ly = max(0.0, min(1.0, lm.y))
        x = int(round(lx * (w - 1)))
        y = int(round(ly * (h - 1)))
        xs.append(x)
        ys.append(y)

    if not xs:
        return None

    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    box_w = max(1, x_max - x_min)
    box_h = max(1, y_max - y_min)

    auto_x_scale = float(box_h / box_w)
    auto_x_scale = max(min_x_scale, min(1.0, auto_x_scale))
    pad_x = int(round(pad_base * auto_x_scale))
    pad_y = pad_base

    x0 = max(0, x_min - pad_x)
    y0 = max(0, y_min - pad_y)
    x1 = min(w, x_max + pad_x + 1)
    y1 = min(h, y_max + pad_y + 1)

    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)

    mask = np.full((h, w), bg, dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    return mask

def create_static_mask(img_size: Tuple[int, int], mask_config: Dict[str, float]) -> np.ndarray:
    """
    Create a static mouth+jaw-focused fallback mask.
    Returns (H, W) float32 mask in [bg..1.0].
    """
    h, w = img_size
    bg = float(mask_config.get("background_mask_value", 0.0))
    # Static fallback ROI tuned for lower-face (mouth+jaw) region.
    # Covers approximately lower-middle 45% of face frame.
    roi_top = int(0.50 * h)
    roi_bottom = int(0.95 * h)
    roi_left = int(0.20 * w)
    roi_right = int(0.80 * w)

    mask = np.full((h, w), bg, dtype=np.float32)
    mask[roi_top:roi_bottom, roi_left:roi_right] = 1.0

    return mask