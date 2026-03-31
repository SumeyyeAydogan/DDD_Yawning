"""
Auto-optimize GradCAM sample weight parameters
with dynamic landmark-based eye+mouth ROI mask.

IMPROVEMENTS:
1. ✅ Uses dynamic landmark mask (MediaPipe FaceMesh) - better than static mask
2. ✅ Better heatmap resize (bilinear interpolation)
3. ✅ Dynamic parameters based on data distribution
4. ✅ Weight analysis with histogram and statistics

Outputs:
- optimized_gradcam_weights.json  (weights per image)
- gradcam_opt_params.json         (best params)
- gradcam_opt_report.txt          (summary report)
- gradcam_weight_histogram.png    (weight distribution)
"""

import os, json, sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from statistics import median
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import mediapipe as mp
import argparse

# Add root path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.gradcam import CustomGradCAM


# ================== CONFIG ======================
CONFIG = {
    "model_path": r"runs/30_epoch_baseline_e3_yawning/models/final_model.h5",
    "data_dir": r"ydd_splitted_dataset/train",
    "img_size": (224, 224),
    "roi_padding_px": 9,          # Padding around the mouth+jaw bounding box (in pixels)
    # When bbox is wide, reduce horizontal padding (pad_x) relative to vertical padding (pad_y).
    # This helps avoid including too much irrelevant horizontal background.
    "roi_keep_aspect_pad_x_min_scale": 0.2,
    "fallback_to_static": True,     # If landmark detection fails, use static mask
    "background_mask_value": 0.2,  # Background value for non-ROI regions (0.0 = hard mask, 0.2 = soft mask)
    "search_level": 2,        # (2 = ORTA)
    "weight_mode": "reward",   # "reward" or "penalize"
    # Weight optimization parameters
    "alpha_max": 3.0,          # Maximum alpha value (was 1.2, now more flexible)
    "alpha_min": 0.3,          # Minimum alpha value (was 0.4)
    "clip_range_factor": 0.8,   # Multiplier for clip range (was 0.6, now more aggressive)
    "weight_max": 5.0,         # Maximum weight value (was 2.0, now allows stronger emphasis)
    "weight_min": 0.1,         # Minimum weight value (was 0.5, now allows more suppression)
    # Alpha evaluation parameters
    "lambda_clip": 1.0,        # Penalty weight for clipping fraction (higher = less clipping preferred)
    "lambda_mean": 0.5,       # Penalty weight for mean deviation from 1.0 (higher = more balanced)
    # "reward": High focus ratio → High weight (reward good behavior)
    # "penalize": Low focus ratio → High weight (penalize bad behavior)
}

# MediaPipe FaceMesh initialization
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)

# Landmark indices for eyes and mouth (MediaPipe FaceMesh)
MOUTH_IDX = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]

# Jaw / chin landmarks
JAW_IDX = [
    152,                 # chin center
    377, 400, 378, 379,  # right jaw
    148, 176, 149, 150   # left jaw
]

ROI_IDX = MOUTH_IDX + JAW_IDX


# ================== MASK ======================
def create_landmark_mask(image_np_uint8, img_size):
    """
    ONE rectangular ROI mask based on min/max of mouth+jaw landmarks (+padding).
    Returns (H,W) float32 mask in [bg..1.0] or None if no face.
    """
    h, w = img_size
    bg = float(CONFIG.get("background_mask_value", 0.0))
    pad_base = int(CONFIG.get("roi_padding_px", 12))

    results = mp_face_mesh.process(image_np_uint8)
    if not results.multi_face_landmarks:
        return None

    face = results.multi_face_landmarks[0]

    xs, ys = [], []
    for i in ROI_IDX:
        lm = face.landmark[i]
        # clamp normalized coords (mediapipe bazen az taşabiliyor)
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

    # bbox çok yatıksa pad_x otomatik küçülür
    min_x_scale = float(CONFIG.get("roi_keep_aspect_pad_x_min_scale", 0.2))
    auto_x_scale = float(box_h / box_w)
    auto_x_scale = max(min_x_scale, min(1.0, auto_x_scale))

    pad_x = int(round(pad_base * auto_x_scale))
    pad_y = pad_base

    x0 = max(0, x_min - pad_x)
    y0 = max(0, y_min - pad_y)

    # IMPORTANT: end-exclusive for slicing
    x1 = min(w, x_max + pad_x + 1)
    y1 = min(h, y_max + pad_y + 1)

    # ensure at least 1px ROI
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)

    mask = np.full((h, w), bg, dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    return mask


def create_static_mask_fallback(img_size):
    """
    Create static mask as fallback when landmark detection fails.
    Uses same logic as src/simple_mask.py.
    """
    h, w = img_size
    background_value = CONFIG.get("background_mask_value", 0.0)
    
    # Define regions (same as simple_mask.py)
    eye_top = int(0.2 * h)
    eye_bottom = int(0.53 * h)
    eye_left = int(0.1 * w)
    eye_right = int(0.9 * w)
    
    mouth_top = int(0.57 * h)
    mouth_bottom = int(0.9 * h)
    mouth_left = int(0.2 * w)
    mouth_right = int(0.8 * w)
    
    # Mask: ROI = 1.0, background = background_value
    mask = np.ones((h, w), dtype=np.float32) * background_value
    
    # Eye region
    mask[eye_top:eye_bottom, eye_left:eye_right] = 1.0
    
    # Mouth region
    mask[mouth_top:mouth_bottom, mouth_left:mouth_right] = 1.0
    
    return mask

def compute_focus_ratio(heatmap, mask):
    """
    Compute focus ratio: how much of the heatmap is in the ROI mask.
    
    Args:
        heatmap: Normalized heatmap (0-1)
        mask: ROI mask (0-1)
    
    Returns:
        Focus ratio (0-1): higher = more focus on ROI
    """
    heatmap = np.maximum(heatmap, 0)  # Ensure non-negative
    if heatmap.max() > 0:
        heatmap = heatmap / (heatmap.max() + 1e-8)  # Normalize to 0-1
    focus = np.sum(heatmap * mask)      # ROI'deki toplam heatmap
    total = np.sum(heatmap) + 1e-8      # Tüm heatmap toplamı
    return float(focus / total)


# ================== CORE ======================
def collect_focus_distribution(model, data_dir, img_size):
    """
    Collect focus ratios for all images using dynamic landmark masks.
    
    IMPROVEMENT: 
    - Uses dynamic landmark mask (MediaPipe) for each image
    - Better heatmap resize with bilinear interpolation
    - Falls back to static mask if landmark detection fails
    """
    gradcam = CustomGradCAM(model)

    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, labels="inferred", label_mode="binary",
        image_size=img_size, batch_size=1, shuffle=False
    )
    file_paths = list(getattr(ds, "file_paths", []))
    path_ds = tf.data.Dataset.from_tensor_slices(file_paths).batch(1)
    ds = tf.data.Dataset.zip((ds, path_ds))
    ds = ds.apply(tf.data.experimental.ignore_errors())

    ratios = []
    landmark_success_count = 0
    static_fallback_count = 0

    print("[AutoOpt] Computing focus distribution...")
    print("[AutoOpt] Using landmark mask with fallback to static (if enabled).")
    print(f"[AutoOpt] Landmark roi-padding: {CONFIG['roi_padding_px']}")
    print(f"[AutoOpt] Fallback to static: {CONFIG['fallback_to_static']}")

    for idx, (data_batch, path_batch) in enumerate(ds):
        images, labels = data_batch
        image = images[0].numpy()
        label = int(labels[0].numpy())
        
        # Convert to uint8 for MediaPipe (0-255 range)
        image_uint8 = (image * 255.0).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        image_normalized = image_uint8 / 255.0 if image.max() > 1.0 else image

        heatmap = gradcam.compute_heatmap(image_normalized, class_idx=label)
        
        # IMPROVEMENT: Better resize with bilinear interpolation
        # Resize heatmap to match image size with better quality
        heatmap = tf.image.resize(
            heatmap[..., None], 
            img_size, 
            method='bilinear', # Better than default
            antialias=True      # Anti-aliasing for smoother resize
        ).numpy()[..., 0]

        # Create dynamic landmark mask for this image (always try landmark; optional static fallback)
        mask = create_landmark_mask(image_uint8, img_size)
        if mask is None:
            if CONFIG['fallback_to_static']:
                mask = create_static_mask_fallback(img_size)
                static_fallback_count += 1
            else:
                print(f"[WARN] No face detected in image {idx+1}, skipping...")
                continue
        else:
            landmark_success_count += 1

        ratio = compute_focus_ratio(heatmap, mask)
        ratios.append(ratio)

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(file_paths)} "
                  f"(landmark: {landmark_success_count}, fallback: {static_fallback_count})")

    print(f"\n[AutoOpt] Summary: {landmark_success_count} landmark masks, "
          f"{static_fallback_count} static fallbacks, {len(ratios)} total processed")

    return ratios, file_paths


# ================== OPTIMIZATION ======================
def choose_params(ratios, level):
    """
    IMPROVEMENT: Dynamic parameter optimization based on data distribution.
    
    Optimizes:
    - target_focus (based on focus distribution with std consideration)
    - alpha (based on virtual penalty sim with more candidates)
    - clip range (scaled from alpha with better bounds)
    """
    ratios = np.array(ratios)
    ratios_sorted = sorted(ratios)
    med = median(ratios)
    mean_r = float(np.mean(ratios))
    std_r = float(np.std(ratios))
    q25 = float(np.percentile(ratios, 25))
    q75 = float(np.percentile(ratios, 75))

    print(f"[AutoOpt] Focus Ratio Stats: mean={mean_r:.3f}, median={med:.3f}, "
          f"std={std_r:.3f}, Q25={q25:.3f}, Q75={q75:.3f}")

    # --------- IMPROVEMENT: Dynamic Target Focus ---------
    # Use median + adaptive offset based on std
    # If std is high, use smaller offset (more conservative)
    # If std is low, use larger offset (more aggressive)
    adaptive_offset = min(0.15, max(0.05, std_r * 0.5))
    target_focus = min(0.92, max(0.45, med + adaptive_offset))
    
    print(f"[AutoOpt] Adaptive offset: {adaptive_offset:.3f}, Target focus: {target_focus:.3f}")

    # --------- IMPROVEMENT: Weight Distribution Based Alpha Optimization ---------
    # Evaluate alpha based on actual weight distribution, not just penalty
    alpha_max = CONFIG.get("alpha_max", 3.0)
    alpha_min = CONFIG.get("alpha_min", 0.3)
    candidates = np.linspace(alpha_min, alpha_max, 20).tolist()  # More granular search
    candidates = [round(c, 2) for c in candidates]  # Round to 2 decimals

    def evaluate_alpha(alpha, ratios, target, cfg):
        """
        Evaluate alpha based on weight distribution quality.
        
        Score = std(weights) - λ_clip * frac_clipped - λ_mean * |mean(weights) - 1|
        
        Higher score = better:
        - High std → good separation between good/bad samples
        - Low clipping → weights stay in natural range
        - Mean ≈ 1 → total loss scaling doesn't change much
        """
        # 1) Calculate clip range from alpha
        clip_range_factor_mult = cfg.get("clip_range_factor", 0.8)
        clip_range_factor = alpha * clip_range_factor_mult
        weight_max = cfg.get("weight_max", 5.0)
        weight_min = cfg.get("weight_min", 0.1)
        clip_min = max(weight_min, 1.0 - clip_range_factor)
        clip_max = min(weight_max, 1.0 + clip_range_factor)
        
        # 2) Calculate weights
        deltas = ratios - target
        weights = 1.0 + alpha * deltas
        weights = np.clip(weights, clip_min, clip_max)
        
        # 3) Calculate metrics
        w_std = float(np.std(weights))
        w_mean = float(np.mean(weights))
        # Fraction of weights that hit the clipping bounds
        frac_clipped = float(np.mean((weights <= clip_min + 1e-6) | (weights >= clip_max - 1e-6)))
        
        # 4) Calculate score with penalties
        lambda_clip = cfg.get("lambda_clip", 1.0)
        lambda_mean = cfg.get("lambda_mean", 0.5)
        
        score = (
            w_std
            - lambda_clip * frac_clipped
            - lambda_mean * abs(w_mean - 1.0)
        )
        
        return score, dict(
            std=w_std,
            mean=w_mean,
            frac_clipped=frac_clipped,
            clip_min=clip_min,
            clip_max=clip_max,
            min_weight=float(np.min(weights)),
            max_weight=float(np.max(weights)),
        )
    
    # Evaluate all candidates
    alpha_scores = {}
    alpha_stats = {}
    for a in candidates:
        score, stats = evaluate_alpha(a, ratios, target_focus, CONFIG)
        alpha_scores[a] = score
        alpha_stats[a] = stats
    
    # Select best alpha (highest score)
    best_alpha = max(alpha_scores, key=alpha_scores.get)
    best_stats = alpha_stats[best_alpha]
    
    print(f"[AutoOpt] Alpha candidates range: [{alpha_min}, {alpha_max}]")
    print(f"[AutoOpt] Best alpha: {best_alpha:.3f} (score: {alpha_scores[best_alpha]:.4f})")
    print(f"[AutoOpt] Best alpha stats: std={best_stats['std']:.3f}, "
          f"mean={best_stats['mean']:.3f}, frac_clipped={best_stats['frac_clipped']:.3f}")
    print(f"[AutoOpt] Best alpha weight range: [{best_stats['min_weight']:.3f}, {best_stats['max_weight']:.3f}]")
    
    # Show top 3 candidates
    top3 = sorted(alpha_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"[AutoOpt] Top 3 alpha candidates:")
    for a, score in top3:
        stats = alpha_stats[a]
        print(f"  Alpha {a:.2f}: score={score:.4f}, std={stats['std']:.3f}, "
              f"clipped={stats['frac_clipped']:.3f}, mean={stats['mean']:.3f}")

    # Clip range is already calculated in evaluate_alpha, use it from best_stats
    clip_min = best_stats['clip_min']
    clip_max = best_stats['clip_max']
    
    print(f"[AutoOpt] Final clip range: [{clip_min:.3f}, {clip_max:.3f}]")

    return dict(
        target_focus=float(target_focus),
        alpha=float(best_alpha),
        clip_min=float(clip_min),
        clip_max=float(clip_max),
        # Additional stats for reporting
        mean_focus=float(mean_r),
        median_focus=float(med),
        std_focus=float(std_r),
        q25_focus=float(q25),
        q75_focus=float(q75),
    )


# ================== APPLY ======================
def apply_weights(params, ratios, file_paths, output_path):
    """
    Apply weights to each image based on focus ratio.
    
    Two modes:
    1. "reward": High focus ratio → High weight (reward good ROI focus)
    2. "penalize": Low focus ratio → High weight (penalize bad ROI focus)
    
    IMPROVEMENT: Returns weight statistics for analysis.
    """
    weights = {}
    weight_values = []
    mode = CONFIG.get("weight_mode", "reward")

    print(f"\n[AutoOpt] Applying weights (mode: {mode})...")

    for r, fp in zip(ratios, file_paths):
        if mode == "reward":
            # Reward approach: High focus ratio → High weight
            # delta = r - target_focus
            # If r > target_focus → delta > 0 → w > 1 (reward)
            # If r < target_focus → delta < 0 → w < 1 (penalize)
            delta = r - params["target_focus"]
        else:  # "penalize"
            # Penalize approach: Low focus ratio → High weight
            # delta = target_focus - r
            # If r < target_focus → delta > 0 → w > 1 (penalize)
            # If r > target_focus → delta < 0 → w < 1 (reward)
            delta = params["target_focus"] - r
        
        w = 1 + params["alpha"] * delta
        w = float(np.clip(w, params["clip_min"], params["clip_max"]))
        
        weight_values.append(w)
        rel = os.path.relpath(fp, CONFIG["data_dir"]).replace("\\", "/")
        weights[rel] = w

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(weights, f, indent=2)

    # Calculate statistics
    weight_stats = {
        "mean": float(np.mean(weight_values)),
        "median": float(np.median(weight_values)),
        "std": float(np.std(weight_values)),
        "min": float(np.min(weight_values)),
        "max": float(np.max(weight_values)),
        "q25": float(np.percentile(weight_values, 25)),
        "q75": float(np.percentile(weight_values, 75)),
    }
    
    print(f"[AutoOpt] Weight Stats: mean={weight_stats['mean']:.3f}, "
          f"median={weight_stats['median']:.3f}, std={weight_stats['std']:.3f}")
    print(f"[AutoOpt] Weight Range: [{weight_stats['min']:.3f}, {weight_stats['max']:.3f}]")

    return weights, weight_values, weight_stats


# ================== ANALYSIS ======================
def plot_weight_histogram(weight_values, ratios, params, output_path):
    """
    IMPROVEMENT: Create histogram and analysis plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Weight distribution histogram
    axes[0, 0].hist(weight_values, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(weight_values), color='r', linestyle='--', label=f'Mean: {np.mean(weight_values):.3f}')
    axes[0, 0].axvline(np.median(weight_values), color='g', linestyle='--', label=f'Median: {np.median(weight_values):.3f}')
    axes[0, 0].set_xlabel('Weight Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Weight Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Focus ratio distribution
    axes[0, 1].hist(ratios, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].axvline(params["target_focus"], color='r', linestyle='--', 
                       label=f'Target: {params["target_focus"]:.3f}')
    axes[0, 1].axvline(np.median(ratios), color='g', linestyle='--', 
                       label=f'Median: {np.median(ratios):.3f}')
    axes[0, 1].set_xlabel('Focus Ratio')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Focus Ratio Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Weight vs Focus Ratio scatter
    axes[1, 0].scatter(ratios, weight_values, alpha=0.3, s=10)
    axes[1, 0].axvline(params["target_focus"], color='r', linestyle='--', 
                       label=f'Target: {params["target_focus"]:.3f}')
    axes[1, 0].set_xlabel('Focus Ratio')
    axes[1, 0].set_ylabel('Weight')
    axes[1, 0].set_title('Weight vs Focus Ratio')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Weight statistics text
    stats_text = f"""
Weight Statistics:
  Mean: {np.mean(weight_values):.3f}
  Median: {np.median(weight_values):.3f}
  Std: {np.std(weight_values):.3f}
  Min: {np.min(weight_values):.3f}
  Max: {np.max(weight_values):.3f}
  Q25: {np.percentile(weight_values, 25):.3f}
  Q75: {np.percentile(weight_values, 75):.3f}

Focus Ratio Statistics:
  Mean: {np.mean(ratios):.3f}
  Median: {np.median(ratios):.3f}
  Std: {np.std(ratios):.3f}
  Target: {params["target_focus"]:.3f}

Parameters:
  Alpha: {params["alpha"]:.3f}
  Clip Range: [{params["clip_min"]:.3f}, {params["clip_max"]:.3f}]
"""
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center', transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[AutoOpt] Histogram saved: {output_path}")


# ================== MAIN ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-optimize GradCAM sample weights and write artifacts."
    )
    parser.add_argument("--model-path", default=CONFIG.get("model_path"))
    parser.add_argument("--data-dir", default=CONFIG.get("data_dir"))
    parser.add_argument("--img-size", default="224,224", help="H,W (e.g. 224,224)")
    parser.add_argument("--search-level", type=int, default=int(CONFIG.get("search_level", 2)))
    parser.add_argument("--weight-mode", choices=["reward", "penalize"], default=CONFIG.get("weight_mode", "reward"))
    parser.add_argument("--roi-padding-px", type=int, default=int(CONFIG.get("roi_padding_px", 12)))
    parser.add_argument("--background-mask-value", type=float, default=float(CONFIG.get("background_mask_value", 0.0)))
    parser.add_argument("--fallback-to-static", type=int, default=1 if CONFIG.get("fallback_to_static", True) else 0)
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory to write outputs (optimized_gradcam_weights.json, gradcam_opt_params.json, report, histogram).",
    )

    args = parser.parse_args()

    h_w = [int(x.strip()) for x in str(args.img_size).split(",") if x.strip()]
    if len(h_w) != 2:
        raise SystemExit("--img-size must be like 224,224")

    cfg = dict(CONFIG)
    cfg["model_path"] = args.model_path
    cfg["data_dir"] = args.data_dir
    cfg["img_size"] = (h_w[0], h_w[1])
    cfg["search_level"] = args.search_level
    cfg["weight_mode"] = args.weight_mode
    cfg["roi_padding_px"] = args.roi_padding_px
    cfg["background_mask_value"] = args.background_mask_value
    cfg["fallback_to_static"] = bool(args.fallback_to_static)

    CONFIG.update(cfg)

    artifacts_dir = args.artifacts_dir
    os.makedirs(artifacts_dir, exist_ok=True)

    model = tf.keras.models.load_model(cfg["model_path"], compile=False)

    ratios, file_paths = collect_focus_distribution(
        model, cfg["data_dir"], cfg["img_size"])

    params = choose_params(ratios, cfg["search_level"])
    print("\n[AutoOpt] Best Params:", {k: v for k, v in params.items() 
                                       if k not in ['mean_focus', 'median_focus', 'std_focus', 'q25_focus', 'q75_focus']})

    # SAVE PARAMS
    with open(os.path.join(artifacts_dir, "gradcam_opt_params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # APPLY & SAVE
    weights, weight_values, weight_stats = apply_weights(
        params, ratios, file_paths,
        output_path=os.path.join(artifacts_dir, "optimized_gradcam_weights.json"))

    # IMPROVEMENT: Create histogram
    plot_weight_histogram(
        weight_values, ratios, params,
        output_path=os.path.join(artifacts_dir, "gradcam_weight_histogram.png")
    )

    # IMPROVEMENT: Enhanced report
    with open(os.path.join(artifacts_dir, "gradcam_opt_report.txt"), "w", encoding="utf-8") as f:
        f.write("=== GradCAM Optimization Report (IMPROVED) ===\n\n")
        f.write("Weight Mode:\n")
        mode = cfg.get("weight_mode", "reward")
        f.write(f"  Mode: {mode}\n")
        if mode == "reward":
            f.write("  Strategy: High focus ratio -> High weight (reward good ROI focus)\n")
        else:
            f.write("  Strategy: Low focus ratio -> High weight (penalize bad ROI focus)\n")
        f.write("\nFocus Ratio Statistics:\n")
        f.write(f"  Mean: {params.get('mean_focus', np.mean(ratios)):.3f}\n")
        f.write(f"  Median: {params.get('median_focus', median(ratios)):.3f}\n")
        f.write(f"  Std: {params.get('std_focus', np.std(ratios)):.3f}\n")
        f.write(f"  Q25: {params.get('q25_focus', np.percentile(ratios, 25)):.3f}\n")
        f.write(f"  Q75: {params.get('q75_focus', np.percentile(ratios, 75)):.3f}\n\n")
        f.write("Optimized Parameters:\n")
        f.write(f"  Target Focus: {params['target_focus']:.3f}\n")
        f.write(f"  Alpha: {params['alpha']:.3f}\n")
        f.write(f"  Clip Range: [{params['clip_min']:.3f}, {params['clip_max']:.3f}]\n\n")
        f.write("Weight Statistics:\n")
        for key, value in weight_stats.items():
            f.write(f"  {key.capitalize()}: {value:.3f}\n")
        f.write(f"\nTotal Samples: {len(ratios)}\n")
        f.write(f"\nMask Configuration:\n")
        f.write(f"  Background Mask Value: {cfg.get('background_mask_value', 0.0):.2f}\n")
        f.write(f"  ROI Padding (mouth+jaw): {cfg['roi_padding_px']}\n")
        f.write(f"  Fallback to Static: {cfg['fallback_to_static']}\n")
        f.write(f"\nWeight Optimization Configuration:\n")
        f.write(f"  Alpha Range: [{cfg.get('alpha_min', 0.3):.2f}, {cfg.get('alpha_max', 3.0):.2f}]\n")
        f.write(f"  Clip Range Factor: {cfg.get('clip_range_factor', 0.8):.2f}\n")
        f.write(f"  Weight Bounds: [{cfg.get('weight_min', 0.1):.2f}, {cfg.get('weight_max', 5.0):.2f}]\n")
        f.write(f"  Lambda Clip (clipping penalty): {cfg.get('lambda_clip', 1.0):.2f}\n")
        f.write(f"  Lambda Mean (mean deviation penalty): {cfg.get('lambda_mean', 0.5):.2f}\n")
        f.write(f"\nAlpha Selection Method:\n")
        f.write(f"  Score = std(weights) - lambda_clip * frac_clipped - lambda_mean * |mean(weights) - 1|\n")
        f.write(f"  Higher score = better separation with balanced distribution\n")

    print(f"\n[AutoOpt] DONE. Files saved in: {artifacts_dir}")
    print("  - optimized_gradcam_weights.json")
    print("  - gradcam_opt_params.json")
    print("  - gradcam_opt_report.txt")
    print("  - gradcam_weight_histogram.png")
