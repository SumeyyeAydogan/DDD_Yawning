import os
import numpy as np
from pathlib import Path
from PIL import Image

import tensorflow as tf
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.gradcam import CustomGradCAM

# ---------------------------------------------------------
# 1) LOAD MODEL (TF-KERAS)
# ---------------------------------------------------------
model1 = tf.keras.models.load_model("runs/30_epoch_baseline_e3_yawning/models/final_model.h5", compile=False)
model2 = tf.keras.models.load_model("runs/30_epoch_exp-reward-soft-e3/models/final_model.h5", compile=False)

# GradCAM objects
cam1 = CustomGradCAM(model1)
cam2 = CustomGradCAM(model2)

# ---------------------------------------------------------
# 2) DIRECTORY SETTINGS
# ---------------------------------------------------------
#project_root = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(project_root, "ydd_splitted_dataset", "test")

classes = ["NoYawn", "Yawn"]

# Output directory
output_dir = Path(project_root) / "combined_fp_to_tn" / "exp-reward-landmark-soft-e3" / "fp_to_tn"
output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 3) IMAGE PREPROCESSING
# ---------------------------------------------------------
IMG_SIZE = (224, 224)

def load_and_preprocess(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    return arr

# ---------------------------------------------------------
# 4) MODEL PREDICTION FUNCTION
# ---------------------------------------------------------
def predict(model, arr):
    p = float(model.predict(arr[np.newaxis, ...], verbose=0)[0][0])
    return 1 if p >= 0.5 else 0

# ---------------------------------------------------------
# 5) FIND FP → TN SAMPLES
# ---------------------------------------------------------
fp_to_tn_paths = []
fn_to_tp_paths = []

for cls in classes:
    cls_dir = os.path.join(test_dir, cls)
    for fname in os.listdir(cls_dir):

        img_path = os.path.join(cls_dir, fname)
        arr = load_and_preprocess(img_path)

        pred1 = predict(model1, arr)
        pred2 = predict(model2, arr)
        true_label = 0 if cls == "NoYawn" else 1

        # FP → TN case:
        # Model1: FP (pred=1, true=0)
        # Model2: TN (pred=0, true=0)
        if true_label == 0 and pred1 == 1 and pred2 == 0:
            fp_to_tn_paths.append(img_path)
        if true_label == 1 and pred1 == 0 and pred2 == 1:
            fn_to_tp_paths.append(img_path)
print("🔍 Number of FP → TN samples:", len(fp_to_tn_paths))
print("🔍 Number of FN → TP samples:", len(fn_to_tp_paths))

""" # ---------------------------------------------------------
# 6) CREATE COMPARATIVE GRADCAM PANEL
# ---------------------------------------------------------
def combine_two_horizontal(img1, img2):
    # Combines two PIL Images side by side.
    w = img1.width + img2.width
    h = max(img1.height, img2.height)
    out = Image.new("RGB", (w, h), "white")
    out.paste(img1, (0, 0))
    out.paste(img2, (img1.width, 0))
    return out


from matplotlib import pyplot as plt

def generate_combined_gradcam(img_path):
    arr = load_and_preprocess(img_path)

    # GradCAM heatmaps
    heat1 = cam1.compute_heatmap(arr, class_idx=1)  # Model1 → FP class 1
    heat2 = cam2.compute_heatmap(arr, class_idx=1)  # Model2 CAM according to class 1 prediction

    # Overlay
    over1 = cam1.overlay_heatmap(heat1, arr)
    over2 = cam2.overlay_heatmap(heat2, arr)

    # NumPy → PIL
    pil1 = Image.fromarray((over1 * 255).astype(np.uint8))
    pil2 = Image.fromarray((over2 * 255).astype(np.uint8))

    combined = combine_two_horizontal(pil1, pil2)
    outname = Path(img_path).stem + "_combined.png"
    combined.save(output_dir / outname)

    print("📁 Saved:", outname) """
# ---------------------------------------------------------
# 6) CREATE COMPARATIVE GRADCAM PANEL (VERTICAL)
# ---------------------------------------------------------
def combine_vertical(img_paths):
    """Verilen dosya yollarını dikey (üst–alt) birleştirir."""
    imgs = [Image.open(p).convert("RGB") for p in img_paths]

    w = max(img.width for img in imgs)
    h = sum(img.height for img in imgs)

    out = Image.new("RGB", (w, h), "white")

    y_offset = 0
    for img in imgs:
        out.paste(img, (0, y_offset))
        y_offset += img.height

    return out


def generate_combined_gradcam_fp_to_tn(img_path):
    """Vertically combine GradCAM outputs of 2 models for each image."""
    arr = load_and_preprocess(img_path)

    # OUTPUT FOLDER AND NAMES
    base_name = Path(img_path).stem
    folder_dir = Path(output_dir) / "fp_to_tn"
    folder_dir.mkdir(parents=True, exist_ok=True)
    out1_path = folder_dir / f"{base_name}_model1.png"
    out2_path = folder_dir / f"{base_name}_model2.png"
    combined_path = folder_dir / f"{base_name}_combined_vertical.png"

    # 1) Save GradCAM for Model1
    cam1.visualize(
        arr,
        class_names=("NoYawn", "Yawn"),
        true_idx=0,   # In the FP→TN scenario, the true label is 0 (NoYawn)
        save_path=str(out1_path)
    )

    # 2) Save GradCAM for Model2
    cam2.visualize(
        arr,
        class_names=("NoYawn", "Yawn"),
        true_idx=0,
        save_path=str(out2_path)
    )

    # 3) Vertically combine
    combined = combine_vertical([out1_path, out2_path])
    combined.save(combined_path)

    print(f"📁 Saved: {combined_path.name}")

def generate_combined_gradcam_fn_to_tp(img_path):
    """Vertically combine GradCAM outputs of 2 models for each image."""
    arr = load_and_preprocess(img_path)

    # OUTPUT FOLDER AND NAMES
    base_name = Path(img_path).stem
    folder_dir = Path(output_dir) / "fn_to_tp"
    folder_dir.mkdir(parents=True, exist_ok=True)
    out1_path = folder_dir / f"{base_name}_model1.png"
    out2_path = folder_dir / f"{base_name}_model2.png"
    combined_path = folder_dir / f"{base_name}_combined_vertical.png"

    # 1) Save GradCAM for Model1
    cam1.visualize(
        arr,
        class_names=("NoYawn", "Yawn"),
        true_idx=1,   # In the FP→TN scenario, the true label is 0 (NoYawn)
        save_path=str(out1_path)
    )

    # 2) Save GradCAM for Model2
    cam2.visualize(
        arr,
        class_names=("NoYawn", "Yawn"),
        true_idx=1,
        save_path=str(out2_path)
    )

    # 3) Vertically combine
    combined = combine_vertical([out1_path, out2_path])
    combined.save(combined_path)

    print(f"📁 Saved: {combined_path.name}")    



# ---------------------------------------------------------
# 7) PROCESS ALL FP→TN SAMPLES
# ---------------------------------------------------------
for p in fp_to_tn_paths:
    generate_combined_gradcam_fp_to_tn(p)
for p in fn_to_tp_paths:
    generate_combined_gradcam_fn_to_tp(p)    

print("\n🎉 All FP→TN GradCAM comparisons have been generated!")
print("📂 Folder:", output_dir)
