"""
Single-purpose script: Black out the bottom-left corner of images with a static
mask and process only a single subject (e.g., "ZC").

No command-line arguments are used; configure paths and parameters below.

What it does
- Scans the test split under DATASET_ROOT for both classes
- Keeps only files whose filename starts with SUBJECT_ONLY (e.g., "ZC_")
- Applies a black rectangle to the bottom-left area with size = RATIO of
  width and height
- Saves masked copies into OUTPUT_DIR, mirroring class subfolders
"""

import os
from typing import List
from PIL import Image
import numpy as np
import tensorflow as tf
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.gradcam import CustomGradCAM
from scripts.combine_images import combine_images


# =========================
# CONFIGURE HERE
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "splitted_dataset")
class_name = "Drowsy"
TEST_DIR = os.path.join(DATASET_ROOT, "test", class_name)
SUBJECT_ONLY = "D" #"m" "zc" "ZC" "D" "d" "h" "w" "M" # Only files beginning with this prefix will be processed  a e u
# "vertical_rectangles"  "original" "bottom_left" "top"
mask_shape = "simple-eye_mouth-visualize"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "masked_test", mask_shape, class_name, f"{SUBJECT_ONLY}")
MAX_IMAGES = 5  # Set to an integer to limit number of processed images

RATIO = 0.35 #0.35  0.20        # Legacy single-ratio (kept for backward compatibility)
WIDTH_RATIO = 0.10    # Horizontal coverage (e.g., 0.10 => %10 of width)
HEIGHT_RATIO = 0.90   # Vertical coverage (e.g., 0.90 => %90 of height)

# Optional: run GradCAM on masked images using an existing trained model
RUN_GRADCAM = True
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "runs",
    "30_epoch_without-mask_sbj-gradcam-fixed",
    "models",
    "final_model.h5",
)
CLASS_NAMES = ("NotDrowsy", "Drowsy")
IMG_SIZE = (224, 224)  # expected model input size (W, H)

if class_name == CLASS_NAMES[0]:
    TRUE_IDX = 0
elif class_name == CLASS_NAMES[1]:
    TRUE_IDX = 1
else:
    TRUE_IDX = 1


def list_images_under(root: str) -> List[str]:
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn.lower())[1] in valid_ext:
                paths.append(os.path.join(dirpath, fn))
    return paths

def apply_top_mask(img: Image.Image, ratio: float) -> Image.Image:
    """Black out two rectangles at bottom-left and bottom-right corners.

    The rectangles cover ratio*width by ratio*height of the image, anchored to
    the two bottom corners.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    #mw = max(1, int(w * ratio))
    mh = max(1, int(h * ratio))
    black = Image.new("RGB", (w, mh), color=(0, 0, 0))
    out = img.copy()
    # bottom-right
    out.paste(black, (0, 0))
    return out

def apply_bottom_left_mask(img: Image.Image, ratio: float) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    mw = max(1, int(w * ratio))
    mh = max(1, int(h * ratio))
    # Create black rectangle and paste at bottom-left (x=0, y=h-mh)
    black = Image.new("RGB", (mw, mh), color=(0, 0, 0))
    out = img.copy()
    out.paste(black, (0, h - mh)) #w-mw
    return out


def apply_bottom_left_right_masks(img: Image.Image, ratio: float) -> Image.Image:
    """Black out two rectangles at bottom-left and bottom-right corners.

    The rectangles cover ratio*width by ratio*height of the image, anchored to
    the two bottom corners.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    mw = max(1, int(w * ratio))
    mh = max(1, int(h * ratio))
    black = Image.new("RGB", (mw, mh), color=(0, 0, 0))
    out = img.copy()
    # bottom-left
    out.paste(black, (0, h - mh))
    # bottom-right
    out.paste(black, (w - mw, h - mh))
    return out


def apply_vertical_rectangles_mask(img: Image.Image, width_ratio: float, height_ratio: float) -> Image.Image:
    """Black out two identical rectangles at bottom-left and bottom-right.

    Each rectangle has size (width_ratio * width) by (height_ratio * height),
    anchored to the bottom corners.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    mw = max(1, int(w * width_ratio))
    mh = max(1, int(h * height_ratio))
    black = Image.new("RGB", (mw, mh), color=(0, 0, 0))
    out = img.copy()
    # bottom-left
    out.paste(black, (0, h - mh))
    # bottom-right
    out.paste(black, (w - mw, h - mh))
    return out


def apply_no_mask(img: Image.Image) -> Image.Image:
    """Return the image unchanged (no mask applied).
    
    This is a passthrough function that ensures RGB mode and returns a copy.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img.copy()


def apply_full_blackout(img: Image.Image) -> Image.Image:
    """Return a completely black image (all pixels set to 0,0,0).
    
    Useful for testing if the model relies on any visual features at all.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    return Image.new("RGB", (w, h), color=(0, 0, 0))

import numpy as np

def visualize_eye_mouth_mask_simple(img: Image.Image, alpha=0.4) -> Image.Image:
    """Overlay eye + mouth ROI using simple paste blocks (no draw)."""
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size

    # ===== ROI Bölge sınırları =====
    eye_top    = int(h * 0.25)
    eye_bottom = int(h * 0.45)

    mouth_top  = int(h * 0.55)
    mouth_bot  = int(h * 0.80)

    # ===== Yarı saydam renk bloğu =====
    overlay_color = (255, 0, 0)
    overlay_eye   = Image.new("RGB", (w, eye_bottom - eye_top), overlay_color)
    overlay_mouth = Image.new("RGB", (w, mouth_bot - mouth_top), overlay_color)

    # ==== TEK KANALLI alpha maskesi =====
    alpha_val = int(alpha * 255)
    alpha_mask_eye   = Image.new("L", (w, eye_bottom - eye_top), alpha_val)
    alpha_mask_mouth = Image.new("L", (w, mouth_bot - mouth_top), alpha_val)

    # ===== Orijinal resmi kopyala =====
    out = img.copy()

    # ===== Doğru maskeyle paste et =====
    out.paste(overlay_eye,   (0, eye_top),   alpha_mask_eye)
    out.paste(overlay_mouth, (0, mouth_top), alpha_mask_mouth)

    return out


def apply_eye_mouth_only_mask(img: Image.Image, use_mean_fill: bool = False, use_soft_mask: bool = False, alpha: float = 0.2) -> Image.Image:
    """Keep only eye and mouth regions visible, black out everything else.
    
    Uses the same regions as SimpleEyeMouthMaskGenerator:
    - Eye region: 20-50% height, 10-90% width
    - Mouth region: 55-90% height, 20-80% width
    
    Args:
        use_mean_fill: If True, fill masked areas with ImageNet mean instead of black
        use_soft_mask: If True, use soft masking (alpha blending) instead of hard mask
        alpha: Transparency for soft masking (0.0 = fully masked, 1.0 = fully visible)
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    
    # Define eye and mouth regions (same as SimpleEyeMouthMaskGenerator)
    eye_top = int(0.2 * h)
    eye_bottom = int(0.53 * h)
    eye_left = int(0.1 * w)
    eye_right = int(0.9 * w)
    
    mouth_top = int(0.57 * h)
    mouth_bottom = int(0.9 * h)
    mouth_left = int(0.2 * w)
    mouth_right = int(0.8 * w)
    
    if use_soft_mask:
        # Soft masking: blend original with masked version
        # Create mask array (1.0 for ROI, alpha for background)
        mask_arr = np.ones((h, w), dtype=np.float32) * alpha
        mask_arr[eye_top:eye_bottom, eye_left:eye_right] = 1.0
        mask_arr[mouth_top:mouth_bottom, mouth_left:mouth_right] = 1.0
        
        # Convert to 3-channel
        mask_3d = np.stack([mask_arr] * 3, axis=-1)
        inv_mask_3d = alpha + (1 - alpha) * (1 - mask_3d)
        
        # Apply soft mask
        img_arr = np.array(img, dtype=np.float32) / 255.0
        out_arr = img_arr * mask_3d
        out_arr_inv = img_arr * inv_mask_3d
        out_arr = np.clip(out_arr * 255.0, 0, 255).astype(np.uint8)
        out_arr_inv = np.clip(out_arr_inv * 255.0, 0, 255).astype(np.uint8)
        #return Image.fromarray(out_arr_inv)
        return Image.fromarray(out_arr)
    
    elif use_mean_fill:
        # Mean fill: use ImageNet mean values for masked areas
        # ImageNet mean (RGB): [0.485, 0.456, 0.406]
        mean_rgb = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        
        # Create mask (1.0 for ROI, 0.0 for masked)
        mask_arr = np.zeros((h, w), dtype=np.float32)
        mask_arr[eye_top:eye_bottom, eye_left:eye_right] = 1.0
        mask_arr[mouth_top:mouth_bottom, mouth_left:mouth_right] = 1.0
        mask_3d = np.stack([mask_arr] * 3, axis=-1)
        
        # Apply: img*M + mean*(1-M)
        img_arr = np.array(img, dtype=np.float32) / 255.0
        out_arr = img_arr * mask_3d + mean_rgb[np.newaxis, np.newaxis, :] * (1.0 - mask_3d)
        out_arr = np.clip(out_arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(out_arr)
    
    else:
        # Hard mask: black background
        out = Image.new("RGB", (w, h), color=(0, 0, 0))
        
        # Copy eye region from original
        eye_region = img.crop((eye_left, eye_top, eye_right, eye_bottom))
        out.paste(eye_region, (eye_left, eye_top))
        
        # Copy mouth region from original
        mouth_region = img.crop((mouth_left, mouth_top, mouth_right, mouth_bottom))
        out.paste(mouth_region, (mouth_left, mouth_top))
        
        return out


def make_out_path(src_path: str) -> str:
    # Mirror structure under TEST_DIR inside OUTPUT_DIR
    rel = os.path.relpath(src_path, start=TEST_DIR)
    out_path = os.path.join(OUTPUT_DIR, rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path


def main() -> None:
    if not os.path.isdir(TEST_DIR):
        raise SystemExit(f"Test directory not found: {TEST_DIR}")

    images = list_images_under(TEST_DIR)
    # Filter by subject prefix on filename
    images = [p for p in images if os.path.basename(p).startswith(SUBJECT_ONLY)]
    if MAX_IMAGES is not None:
        images = images[:MAX_IMAGES]
    if not images:
        print(f"No images starting with '{SUBJECT_ONLY}' found under: {TEST_DIR}")
        return

    print(f"Found {len(images)} images for subject '{SUBJECT_ONLY}'.")
    print(f"Writing masked images under: {OUTPUT_DIR}")

    cam = None
    if RUN_GRADCAM:
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            """
            # Create custom loss function for loading
            loss_fn = create_simple_masked_loss()
            
            # Load model with custom objects
            model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={'loss_fn': loss_fn}
            )
            """
            cam = CustomGradCAM(model)
        except Exception as e:
            print(f"GradCAM disabled (model load failed): {e}")
            cam = None
    gradcam_root = OUTPUT_DIR + "_gradcam"

    for idx, path in enumerate(images, 1):
        try:
            img = Image.open(path)
            # Use two-rectangle bottom corners mask with different ratios
            # For testing: change to apply_full_blackout(img) to test with completely black image
            masked = visualize_eye_mouth_mask_simple(img, alpha=0.7)
            #apply_eye_mouth_only_mask(img, use_soft_mask=True, alpha=0.2)
            #apply_vertical_rectangles_mask(img, WIDTH_RATIO, HEIGHT_RATIO) apply_no_mask(img) apply_bottom_left_mask(img, RATIO)
            #apply_eye_mouth_only_mask(img)
            out_path = make_out_path(path)
            masked.save(out_path)
            # Also run GradCAM on the masked image if enabled
            if cam is not None:
                # Resize to model input size to avoid shape mismatch
                resized = masked.resize(IMG_SIZE, Image.BILINEAR)
                arr = np.array(resized).astype("float32") / 255.0

                # DEBUG: Verify we're using masked image and check prediction
                if idx <= 3:
                    # Direct prediction on masked image for debugging
                    model = cam.original_model
                    pred_masked = model.predict(arr[None, ...], verbose=0)
                    prob_masked_drowsy = float(pred_masked[0][0])  # P(Drowsy)
                    prob_masked_notdrowsy = 1.0 - prob_masked_drowsy  # P(NotDrowsy)
                    pred_cls_masked = 1 if prob_masked_drowsy >= 0.5 else 0
                    
                    # Also check original image for comparison (for logging)
                    orig_resized_dbg = img.resize(IMG_SIZE, Image.BILINEAR)
                    arr_orig_dbg = np.array(orig_resized_dbg).astype("float32") / 255.0
                    pred_orig = model.predict(arr_orig_dbg[None, ...], verbose=0)
                    prob_orig_drowsy = float(pred_orig[0][0])  # P(Drowsy)
                    prob_orig_notdrowsy = 1.0 - prob_orig_drowsy  # P(NotDrowsy)
                    pred_cls_orig = 1 if prob_orig_drowsy >= 0.5 else 0
                    
                    # Test with different values: white, gray
                    arr_white = np.ones((224, 224, 3), dtype=np.float32)
                    pred_white = model.predict(arr_white[None, ...], verbose=0)
                    prob_white = float(pred_white[0][0])
                    
                    arr_gray = np.ones((224, 224, 3), dtype=np.float32) * 0.5
                    pred_gray = model.predict(arr_gray[None, ...], verbose=0)
                    prob_gray = float(pred_gray[0][0])
                    
                    print(f"  Masked image shape: {arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
                
                # Prepare GradCAM output paths (orig, masked, combined)
                rel_path = os.path.relpath(out_path, start=OUTPUT_DIR)
                base_name, _ = os.path.splitext(os.path.basename(rel_path))
                gc_dir = os.path.join(gradcam_root, os.path.dirname(rel_path))
                # Subfolders
                orig_dir = os.path.join(gc_dir, "original")
                masked_dir = os.path.join(gc_dir, "masked")
                combined_dir = os.path.join(gc_dir, "combined")
                os.makedirs(orig_dir, exist_ok=True)
                os.makedirs(masked_dir, exist_ok=True)
                os.makedirs(combined_dir, exist_ok=True)

                orig_gc_out = os.path.join(orig_dir, f"{base_name}_orig.png")
                masked_gc_out = os.path.join(masked_dir, f"{base_name}_masked.png")
                combined_gc_out = os.path.join(combined_dir, f"{base_name}_combined.png")

                # Build original array for fair comparison
                orig_resized = img.resize(IMG_SIZE, Image.BILINEAR)
                arr_orig = np.array(orig_resized).astype("float32") / 255.0

                # Save GradCAMs
                cam.visualize(arr_orig, class_names=CLASS_NAMES, true_idx=TRUE_IDX, save_path=orig_gc_out)
                cam.visualize(arr, class_names=CLASS_NAMES, true_idx=TRUE_IDX, save_path=masked_gc_out)

                # Combine and save
                try:
                    combined_img = combine_images([orig_gc_out, masked_gc_out], layout="vertical")
                    combined_img.save(combined_gc_out)
                except Exception as e:
                    print(f"Combine failed for {combined_gc_out}: {e}")
            if idx <= 5:
                print(f"Saved -> {out_path}")
        except Exception as e:
            print(f"Failed: {path} -> {e}")

    print("Done.")


if __name__ == "__main__":
    main()