import os, json, sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

# Add root path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.gradcam import CustomGradCAM
from src.analysis_pipeline import get_analysis_pipeline_from_data_dir
from src.mask_helpers import create_landmark_mask
from src.focus_metrics import compute_focus_ratio


# ================== CONFIG ======================
CONFIG = {
    #"model_path": r"runs/30_epoch_exp-reward-mouth-jaw-11-landmark/models/final_model.h5",
    "model_path": r"runs/30_epoch_baseline_e3_yawning/models/final_model.h5",
    "data_dir": r"ydd_splitted_dataset/test",  # Changed to test dataset
    "img_size": (224, 224),
    "model_name": "original model",  # Model name for histogram title
    "dataset_name": "test",  # Dataset name for histogram title
    "use_landmark_mask": True,  # Use dynamic landmark mask
    #"landmark_box_half_size": 15,  # Half side-length of square patches around landmarks
    "roi_padding_px": 6,
    "background_mask_value": 0.2,  # Background value for non-ROI regions (0.0 = hard mask, 0.2 = soft mask)
}

# ================== CORE ======================
def collect_focus_distribution_with_predictions(model, data_dir, img_size):
    """
    Collect focus ratios and model predictions for all images using dynamic landmark masks.
    
    Returns:
        y_true: List of true labels
        y_pred: List of predicted labels (0 or 1)
        y_prob: List of prediction probabilities
        focus_ratios: List of focus ratios
    """
    gradcam = CustomGradCAM(model)

    ds, file_paths = get_analysis_pipeline_from_data_dir(data_dir, img_size)

    y_true = []
    y_pred = []
    y_prob = []
    focus_ratios = []
    landmark_success_count = 0

    print("[Statistics] Computing focus distribution with predictions...")
    print("[Statistics] Using landmark mask")
    print(f"[Statistics] Landmark box half-size: {CONFIG['roi_padding_px']}")

    for idx, (data_batch, path_batch) in enumerate(ds):
        images, labels = data_batch
        image = images[0].numpy()
        label = int(labels[0].numpy())
        
        # Convert to uint8 for MediaPipe (0-255 range)
        image_uint8 = (image * 255.0).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        image_normalized = image_uint8 / 255.0 if image.max() > 1.0 else image

        # Get model prediction
        preds = model.predict(image_normalized[None, ...], verbose=0)
        prob = float(preds[0][0])
        pred = 1 if prob >= 0.5 else 0
        
        # Use predicted class for GradCAM
        class_idx = pred
        heatmap = gradcam.compute_heatmap(image_normalized, class_idx=class_idx)
        
        # Resize heatmap to match image size with bilinear interpolation
        heatmap = tf.image.resize(
            heatmap[..., None], 
            img_size, 
            method='bilinear',
            antialias=True
        ).numpy()[..., 0]

        # Create dynamic landmark mask for this image
        mask = None
        if CONFIG['use_landmark_mask']:
            mask = create_landmark_mask(image_uint8, img_size, CONFIG)
            if mask is not None:
                landmark_success_count += 1
        
        # If no valid mask, skip this image
        if mask is None:
            print(f"[WARN] No face detected in image {idx+1}, skipping...")
            continue

        ratio = compute_focus_ratio(heatmap, mask)
        
        y_true.append(label)
        y_pred.append(pred)
        y_prob.append(prob)
        focus_ratios.append(ratio)

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(file_paths)} "
                  f"(landmark: {landmark_success_count})")

    print(f"\n[Statistics] Summary: {landmark_success_count} landmark masks, "
          f"{len(focus_ratios)} total processed")

    return y_true, y_pred, y_prob, focus_ratios

# ================== ANALYSIS ======================
def get_confusion_matrix_groups(y_true, y_pred):
    """
    Calculate TN, TP, FN, FP groups from true and predicted labels.
    
    Returns:
        Dictionary with keys: 'TN', 'TP', 'FN', 'FP' containing indices
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    groups = {
        'TN': np.where((y_true == 0) & (y_pred == 0))[0],
        'TP': np.where((y_true == 1) & (y_pred == 1))[0],
        'FN': np.where((y_true == 1) & (y_pred == 0))[0],
        'FP': np.where((y_true == 0) & (y_pred == 1))[0],
    }
    
    return groups

def plot_focus_ratio_by_confusion_matrix(y_true, y_pred, focus_ratios, model_name, dataset_name, output_path):
    """
    Plot focus ratio distributions for TN, TP, FN, FP groups in separate subplots.
    Uses shared/global bins + shared x/y-axis limits for apples-to-apples comparison.
    """
    groups = get_confusion_matrix_groups(y_true, y_pred)
    focus_ratios = np.array(focus_ratios, dtype=np.float32)

    # ---- Guard: no samples ----
    if focus_ratios.size == 0:
        print("[WARN] No focus ratios to plot. Skipping histogram.")
        return

    # ---- Shared X range ----
    # Focus ratio theoretically [0, 1]. Best comparison:
    x_min, x_max = 0.0, 1.0

    # ---- Shared bins (same edges everywhere) ----
    NBINS = 50
    bins = np.linspace(x_min, x_max, NBINS + 1)

    # ---- Compute global max density for shared ylim ----
    global_max_density = 0.0
    for g in ['TN', 'TP', 'FN', 'FP']:
        idxs = groups[g]
        if len(idxs) == 0:
            continue
        vals = focus_ratios[idxs]

        # Clamp (numerical overflow etc.)
        vals = np.clip(vals, x_min, x_max)

        hist, _ = np.histogram(vals, bins=bins, density=True)
        if hist.size > 0:
            global_max_density = max(global_max_density, float(hist.max()))

    if global_max_density <= 0:
        global_max_density = 1.0
    y_max = global_max_density * 1.05

    # ---- Figure ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Focus Ratio Distribution - {model_name} - {dataset_name}',
                 fontsize=16, fontweight='bold')

    colors = {'TN': 'green', 'TP': 'blue', 'FN': 'red', 'FP': 'orange'}
    group_order = [('TN', 0, 0), ('TP', 0, 1), ('FN', 1, 0), ('FP', 1, 1)]

    for group_name, row, col in group_order:
        ax = axes[row, col]
        indices = groups[group_name]

        # Shared x/y-axis limits (for each subplot)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max)

        if len(indices) > 0:
            group_ratios = focus_ratios[indices]
            group_ratios = np.clip(group_ratios, x_min, x_max)

            ax.hist(
                group_ratios,
                bins=bins,                # <-- shared bins
                edgecolor='black',
                alpha=0.7,
                color=colors[group_name],
                density=True
            )

            median_val = float(np.median(group_ratios))
            mean_val = float(np.mean(group_ratios))

            ax.axvline(median_val, color='red', linestyle='--', linewidth=2,
                       label=f'Median: {median_val:.3f}')
            ax.axvline(mean_val, color='blue', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_val:.3f}')

            ax.set_title(f'Focus Ratio Distribution ({group_name})\nCount: {len(indices)}',
                         fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            stats_text = (
                f"Mean: {mean_val:.3f}\n"
                f"Median: {median_val:.3f}\n"
                f"Std: {float(np.std(group_ratios)):.3f}\n"
                f"Min: {float(np.min(group_ratios)):.3f}\n"
                f"Max: {float(np.max(group_ratios)):.3f}"
            )
            ax.text(
                0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
        else:
            ax.text(0.5, 0.5, f'No {group_name} samples',
                    transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title(f'Focus Ratio Distribution ({group_name})\nCount: 0',
                         fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)

        ax.set_xlabel('Focus Ratio', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Statistics] Histogram saved: {output_path}")

  

# ================== MAIN ======================
if __name__ == "__main__":
    cfg = CONFIG

    # Load model
    model = tf.keras.models.load_model(cfg["model_path"], compile=False)
    
    # Extract model name from path (optional, can be overridden)
    model_name = cfg.get("model_name", Path(cfg["model_path"]).parent.parent.name)
    dataset_name = cfg.get("dataset_name", Path(cfg["data_dir"]).name)

    # Collect focus ratios with predictions
    y_true, y_pred, y_prob, focus_ratios = collect_focus_distribution_with_predictions(
        model, cfg["data_dir"], cfg["img_size"])

    # Calculate confusion matrix groups
    groups = get_confusion_matrix_groups(y_true, y_pred)
    
    print("\n[Statistics] Confusion Matrix Summary:")
    print(f"  TN (True Negative): {len(groups['TN'])}")
    print(f"  TP (True Positive): {len(groups['TP'])}")
    print(f"  FN (False Negative): {len(groups['FN'])}")
    print(f"  FP (False Positive): {len(groups['FP'])}")
    print(f"  Total: {len(y_true)}")
    
    # Print focus ratio statistics for each group
    print("\n[Statistics] Focus Ratio Statistics by Group:")
    for group_name in ['TN', 'TP', 'FN', 'FP']:
        indices = groups[group_name]
        if len(indices) > 0:
            group_ratios = np.array(focus_ratios)[indices]
            print(f"  {group_name}: Mean={np.mean(group_ratios):.3f}, "
                  f"Median={np.median(group_ratios):.3f}, "
                  f"Std={np.std(group_ratios):.3f}, Count={len(indices)}")
        else:
            print(f"  {group_name}: No samples")

    # Create output directory
    os.makedirs("artifacts", exist_ok=True)
    
    # Generate histogram filename: model_name - dataset_name
    histogram_filename = f"{model_name} - {dataset_name}.png"
    output_path = os.path.join("artifacts", histogram_filename)
    
    # Plot focus ratio distributions by confusion matrix groups
    plot_focus_ratio_by_confusion_matrix(
        y_true, y_pred, focus_ratios, 
        model_name, dataset_name, 
        output_path
    )

    print(f"\n[Statistics] DONE. Histogram saved: {output_path}")