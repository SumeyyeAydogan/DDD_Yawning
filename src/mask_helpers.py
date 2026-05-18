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


def create_landmark_mask(
    image_np_uint8: np.ndarray,
    img_size: Tuple[int, int],
    mask_config: Dict[str, float],
) -> Optional[np.ndarray]:
    """
    Build one rectangular ROI mask based on mouth+jaw landmarks (+padding).
    Returns (H, W) float32 mask in [bg..1.0] or None if no face.
    """
    h, w = img_size
    bg = float(mask_config.get("background_mask_value", 0.0))
    pad_base = int(mask_config.get("roi_padding_px", 12))
    min_x_scale = float(mask_config.get("roi_keep_aspect_pad_x_min_scale", 0.2))

    results = _face_mesh.process(image_np_uint8)
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