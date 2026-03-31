import os, sys, json
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image, ImageDraw
import mediapipe as mp

# Add root path (project structure assumed same as before)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.gradcam import CustomGradCAM


# ================== CONFIG (YAWNING) ======================
CONFIG = {
    # Yawning dataset path:
    "data_dir": r"ydd_splitted_dataset/test",   # <-- change if needed
    "img_size": (224, 224),
    "dataset_name": "test",

    # TF dataset class order for yawning:
    # NoYawn = 0, Yawn = 1
    "class_names": ["NoYawn", "Yawn"],

    # Mask params
    "landmark_box_half_size": 15,
    "background_mask_value": 0.2,   # 0.0 = hard mask, 0.2 = soft mask
    "roi_padding_px": 11,
    # When bbox is wide, reduce horizontal padding (pad_x) relative to vertical padding (pad_y).
    "roi_keep_aspect_pad_x_min_scale": 0.2,

    # Threshold rule
    "threshold_source": "baseline_median",

    # Plot
    "hist_bins": 50,
}

# 4 models (yawning)
# IMPORTANT: "original" must exist for threshold
# MODEL_CONFIGS = [
#     {"label": "original",   "model_path": r"runs/30_epoch_baseline_e3_yawning/models/final_model.h5"},
#     {"label": "reward",     "model_path": r"runs/mouth-mask/landmark-15/30_epoch_reward-e3/models/final_model.h5"},
#     {"label": "log-reward", "model_path": r"runs/mouth-mask/landmark-15/30_epoch_log-reward-e3/models/final_model.h5"},
#     {"label": "exp-reward", "model_path": r"runs/mouth-mask/landmark-15/30_epoch_exp-reward-e3/models/final_model.h5"},
# ]
# MODEL_CONFIGS = [
#     {"label": "original",   "model_path": r"runs/30_epoch_baseline_e3_yawning/models/final_model.h5"},
#     {"label": "reward",     "model_path": r"runs/30_epoch_reward-e3/models/final_model.h5"},
#     {"label": "log-reward", "model_path": r"runs/30_epoch_log-reward-e3/models/final_model.h5"},
#     {"label": "exp-reward", "model_path": r"runs/30_epoch_exp-reward-e3/models/final_model.h5"},
# ]
MODEL_CONFIGS = [
    {"label": "original",   "model_path": r"runs/30_epoch_baseline_e3_yawning/models/final_model.h5"},
    {"label": "reward",     "model_path": r"runs/30_epoch_reward-mouth-jaw-10-landmark/models/final_model.h5"},
    {"label": "log-reward", "model_path": r"runs/30_epoch_log-reward-mouth-jaw-10-landmark/models/final_model.h5"},
    {"label": "exp-reward", "model_path": r"runs/30_epoch_exp-reward-mouth-jaw-9-landmark/models/final_model.h5"},
]

# ================== MediaPipe FaceMesh ======================
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)

# Landmark indices (MediaPipe FaceMesh)
# For yawning, mouth ROI is most relevant. Keep only MOUTH by default.
#MOUTH_IDX = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]

# If you want eyes + mouth, uncomment these and use ROI_IDX = LEFT+RIGHT+MOUTH
# LEFT_EYE_IDX  = [33, 7, 163, 144, 145, 153, 154, 155, 133]
# RIGHT_EYE_IDX = [263, 249, 390, 373, 374, 380, 381, 382, 362]
# ROI_IDX = LEFT_EYE_IDX + RIGHT_EYE_IDX + MOUTH_IDX
# Mouth landmarks
MOUTH_IDX = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]

# Jaw / chin landmarks
JAW_IDX = [
    152,                 # chin center
    377, 400, 378, 379,  # right jaw
    148, 176, 149, 150   # left jaw
]

ROI_IDX = MOUTH_IDX + JAW_IDX


#================== MASK + FOCUS ======================
# def create_landmark_mask(image_np_uint8, img_size):
#     """
#     image_np_uint8: (H,W,3) RGB uint8 at img_size
#     returns: (H,W) float32 mask in [0,1] or None if no face
#     """
#     h, w = img_size
#     background_value = float(CONFIG.get("background_mask_value", 0.0))

#     results = mp_face_mesh.process(image_np_uint8)
#     if not results.multi_face_landmarks:
#         return None

#     face = results.multi_face_landmarks[0]

#     bg_pil_value = int(background_value * 255)
#     pil_mask = Image.new("L", (w, h), bg_pil_value)
#     draw = ImageDraw.Draw(pil_mask)

#     box_half_size = int(CONFIG["landmark_box_half_size"])
#     for i in MOUTH_IDX:
#         lm = face.landmark[i]
#         cx = int(lm.x * w)
#         cy = int(lm.y * h)
#         x0 = max(0, cx - box_half_size)
#         y0 = max(0, cy - box_half_size)
#         x1 = min(w - 1, cx + box_half_size)
#         y1 = min(h - 1, cy + box_half_size)
#         draw.rectangle([x0, y0, x1, y1], outline=255, fill=255)

#     return np.array(pil_mask, dtype=np.float32) / 255.0
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




def compute_focus_ratio(heatmap, mask):
    heatmap = np.maximum(heatmap, 0)
    mx = float(heatmap.max())
    if mx > 0:
        heatmap = heatmap / (mx + 1e-8)
    focus = float(np.sum(heatmap * mask))
    total = float(np.sum(heatmap) + 1e-8)
    return float(focus / total)


# ================== CORE ======================
def collect_focus_ratios(model, data_dir, img_size, class_names):
    """
    Returns:
      focus_ratios: np.array (only face_ok==1)
      stats: dict with N_total, N_face, face_rate
    """
    gradcam = CustomGradCAM(model)

    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        class_names=class_names,  # yawning: ["NoYawn","Yawn"]
        image_size=img_size,
        batch_size=1,
        shuffle=False
    )
    file_paths = list(getattr(ds, "file_paths", []))
    path_ds = tf.data.Dataset.from_tensor_slices(file_paths).batch(1)
    ds = tf.data.Dataset.zip((ds, path_ds))
    ds = ds.apply(tf.data.experimental.ignore_errors())

    ratios = []
    face_ok = 0
    total = 0

    for idx, (data_batch, path_batch) in enumerate(ds):
        total += 1
        images, labels = data_batch
        image = images[0].numpy()

        # For MediaPipe: uint8 0..255
        image_uint8 = (image * 255.0).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        # For model: float 0..1
        image_norm = image_uint8 / 255.0

        # prediction -> class_idx for gradcam
        prob = float(model.predict(image_norm[None, ...], verbose=0)[0][0])
        pred = 1 if prob >= 0.5 else 0

        # GradCAM heatmap (pred class)
        heatmap = gradcam.compute_heatmap(image_norm, class_idx=pred)
        heatmap = tf.image.resize(
            heatmap[..., None],
            img_size,
            method="bilinear",
            antialias=True
        ).numpy()[..., 0]

        mask = create_landmark_mask(image_uint8, img_size)
        if mask is not None:
            face_ok += 1
            ratios.append(compute_focus_ratio(heatmap, mask))

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx+1}/{len(file_paths)} | face_ok={face_ok}")

    ratios = np.array(ratios, dtype=np.float32)
    stats = {
        "N_total": int(total),
        "N_face": int(face_ok),
        "face_rate": float(face_ok / max(total, 1))
    }
    return ratios, stats


# ================== PLOT ======================
# def plot_histograms(results_dict, dataset_name, out_path):
#     labels = list(results_dict.keys())
#     labels = [l for l in labels if len(results_dict[l]) > 0]
#     if not labels:
#         print("[WARN] No results to plot.")
#         return

#     all_vals = np.concatenate([results_dict[l] for l in labels])
#     if len(all_vals) == 0:
#         print("[WARN] Empty arrays, skip plot.")
#         return

#     bins = int(CONFIG.get("hist_bins", 50))
#     x_min, x_max = float(all_vals.min()), float(all_vals.max())
#     bin_edges = np.linspace(x_min, x_max, bins + 1)

#     fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4), squeeze=False)
#     fig.suptitle(f"Focus Ratio Distributions - {dataset_name}", fontsize=14, fontweight="bold")

#     for i, label in enumerate(labels):
#         ax = axes[0, i]
#         vals = results_dict[label]
#         ax.hist(vals, bins=bin_edges, density=True, edgecolor="black", alpha=0.75)
#         ax.set_title(f"{label} (n={len(vals)})")
#         ax.set_xlabel("Focus Ratio")
#         ax.set_ylabel("Density")
#         ax.grid(True, alpha=0.25)
#         ax.axvline(np.mean(vals), linestyle="--", linewidth=2, label=f"Mean {np.mean(vals):.3f}")
#         ax.axvline(np.median(vals), linestyle="--", linewidth=2, label=f"Median {np.median(vals):.3f}")
#         ax.legend(fontsize=9)

#     plt.tight_layout()
#     plt.savefig(out_path, dpi=300, bbox_inches="tight")
#     plt.close()
#     print(f"[SAVE] {out_path}")
def plot_focus_ratio_by_model(results_dict, dataset_name, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Focus Ratio Distribution by Model - {dataset_name}', fontsize=16, fontweight='bold')

    order = [
        ("original", 0, 0),
        ("reward", 0, 1),
        ("log-reward", 1, 0),
        ("exp-reward", 1, 1),
    ]

    # ---------- 1) GLOBAL X RANGE ----------
    nonempty = [v for v in results_dict.values() if v is not None and len(v) > 0]
    all_vals = np.concatenate(nonempty) if len(nonempty) > 0 else np.array([0.0, 1.0])

    x_min = float(np.min(all_vals))
    x_max = float(np.max(all_vals))

    # İstersen sabitle:
    # x_min, x_max = 0.0, 1.0

    # ---------- 2) SAME BINS ----------
    n_bins = int(CONFIG.get("hist_bins", 50))
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)

    # ---------- 3) GLOBAL Y RANGE ----------
    global_ymax = 0.0
    for label, _, _ in order:
        vals = results_dict.get(label, np.array([]))
        if vals is None or len(vals) == 0:
            continue
        hist, _ = np.histogram(vals, bins=bin_edges, density=True)
        global_ymax = max(global_ymax, float(hist.max()))
    global_ymax *= 1.10 if global_ymax > 0 else 1.0

    colors = {
        "original": "blue",
        "reward": "green",
        "log-reward": "orange",
        "exp-reward": "red",
    }

    for label, row, col in order:
        ax = axes[row, col]
        ratios = results_dict.get(label, np.array([]))
        if ratios is None:
            ratios = np.array([])

        if len(ratios) > 0:
            ax.hist(
                ratios,
                bins=bin_edges,
                edgecolor='black',
                alpha=0.7,
                color=colors.get(label, "gray"),
                density=True
            )

            median_val = float(np.median(ratios))
            mean_val = float(np.mean(ratios))

            ax.axvline(median_val, color='red', linestyle='--', linewidth=2,
                       label=f'Median: {median_val:.3f}')
            ax.axvline(mean_val, color='black', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_val:.3f}')

            ax.set_title(f'{label} (Count: {len(ratios)})', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            stats_text = (
                f'Mean: {mean_val:.3f}\n'
                f'Median: {median_val:.3f}\n'
                f'Std: {np.std(ratios):.3f}\n'
                f'Min: {np.min(ratios):.3f}\n'
                f'Max: {np.max(ratios):.3f}'
            )
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, f'No data for {label}',
                    transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title(label, fontsize=13, fontweight='bold')

        # ---------- 4) FORCE SAME AXES ----------
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, global_ymax)
        ax.set_xlabel('Focus Ratio', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Model Comparison] Histogram saved: {output_path}")


# ================== SUMMARY SAVE ======================
def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
def print_summary_table(rows, dataset_name):
    """
    rows: list of dicts with keys:
      model_label, N_face, N_total, threshold_T, P_focus_above_T, delta_P_vs_baseline,
      mean_focus, median_focus, std_focus
    """
    print("\n================= SUMMARY TABLE =================")
    print(f"Dataset: {dataset_name}")
    headers = ["Model", "N_face", "N_total", "T", "P(focus>T)", "ΔP", "Mean", "Median", "Std"]
    colw = [10, 7, 7, 7, 11, 7, 7, 7, 7]

    def fmt_row(vals):
        return "  ".join(str(v).ljust(w) for v, w in zip(vals, colw))

    print(fmt_row(headers))
    print(fmt_row(["-"*len(h) for h in headers]))

    for r in rows:
        vals = [
            r["model_label"],
            r["N_face"],
            r["N_total"],
            f'{r["threshold_T"]:.4f}',
            f'{r["P_focus_above_T"]:.4f}',
            f'{r["delta_P_vs_baseline"]:+.4f}',
            f'{r["mean_focus"]:.4f}',
            f'{r["median_focus"]:.4f}',
            f'{r["std_focus"]:.4f}',
        ]
        print(fmt_row(vals))

def print_summary_table_markdown(rows, dataset_name):
    """
    rows: list of summary dicts
    Prints a GitHub/Markdown-compatible table.
    """
    print("\n================= SUMMARY TABLE (MARKDOWN) =================")
    print(f"**Dataset:** `{dataset_name}`\n")

    headers = [
        "Model", "N_face", "N_total", "T",
        "P(focus > T)", "ΔP vs baseline",
        "Mean", "Median", "Std"
    ]

    # header
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")

    for r in rows:
        print(
            "| "
            f"{r['model_label']} | "
            f"{r['N_face']} | "
            f"{r['N_total']} | "
            f"{r['threshold_T']:.4f} | "
            f"{r['P_focus_above_T']:.4f} | "
            f"{r['delta_P_vs_baseline']:+.4f} | "
            f"{r['mean_focus']:.4f} | "
            f"{r['median_focus']:.4f} | "
            f"{r['std_focus']:.4f} |"
        )

# ================== HISTOGRAM-INTEGRAL (NEW) ======================
def histogram_right_tail_area(ratios, T, bin_edges):
    """
    Approximates integral of the estimated PDF from T to +inf using histogram bins:
      area ≈ sum_{bins with center >= T} (density_bin * bin_width)

    Requires density=True histogram and a shared bin_edges for fair comparison.
    """
    ratios = np.asarray(ratios, dtype=np.float32)
    if ratios.size == 0:
        return 0.0
    hist, edges = np.histogram(ratios, bins=bin_edges, density=True)
    widths = np.diff(edges)
    centers = (edges[:-1] + edges[1:]) * 0.5
    area_bins = hist * widths
    return float(np.sum(area_bins[centers >= T]))


# ================== MAIN ======================
if __name__ == "__main__":
    cfg = CONFIG
    data_dir = cfg["data_dir"]
    img_size = tuple(cfg["img_size"])
    dataset_name = cfg.get("dataset_name", Path(data_dir).name)

    os.makedirs("artifacts", exist_ok=True)
    summary_path = os.path.join("artifacts", f"focus_summary_{dataset_name}.jsonl")

    # 1) Collect ratios per model
    ratios_by_model = {}
    stats_by_model = {}
    model_paths = {}

    for m_cfg in MODEL_CONFIGS:
        label = m_cfg["label"]
        model_path = m_cfg["model_path"]
        model_paths[label] = model_path

        print(f"\n[LOAD] {label}: {model_path}")
        if not os.path.exists(model_path):
            print(f"[WARN] Missing model path, skip: {model_path}")
            continue

        model = tf.keras.models.load_model(model_path, compile=False)

        print(f"[RUN] Collecting focus ratios for {label} ...")
        ratios, stats = collect_focus_ratios(model, data_dir, img_size, cfg["class_names"])

        ratios_by_model[label] = ratios
        stats_by_model[label] = stats

        print(f"[DONE] {label}: N_face={stats['N_face']} / N_total={stats['N_total']} | mean={ratios.mean():.4f} | median={np.median(ratios):.4f}")

    if "original" not in ratios_by_model or len(ratios_by_model["original"]) == 0:
        raise RuntimeError("Baseline 'original' ratios missing/empty. Cannot compute threshold.")

    # 2) Threshold from baseline (median)
    T = float(np.median(ratios_by_model["original"]))
    print("\n================= THRESHOLD =================")
    print(f"Threshold source: baseline median")
    print(f"T = median(focus_original) = {T:.4f}")

    # 3) Compute integral metric and write summaries
    print("\n================= INTEGRAL METRIC =================")
    # "Integral" == P(focus > T)
    # Baseline P(focus > T)
    baseline_ratios = ratios_by_model["original"]
    P_baseline = float(np.mean(baseline_ratios > T))

    # (NEW) True histogram-integral style right-tail area with shared bins
    nonempty = [v for v in ratios_by_model.values() if v is not None and len(v) > 0]
    all_vals = np.concatenate(nonempty) if len(nonempty) > 0 else np.array([0.0, 1.0], dtype=np.float32)
    x_min = float(np.min(all_vals))
    x_max = float(np.max(all_vals))
    n_bins = int(cfg.get("hist_bins", 50))
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    A_hist_baseline = histogram_right_tail_area(baseline_ratios, T, bin_edges)

    summary_rows = []

    for label, ratios in ratios_by_model.items():
        if len(ratios) == 0:
            continue

        p_above = float(np.mean(ratios > T))
        delta_p = p_above - P_baseline   # <<< YENİ METRİK

        # (NEW) histogram-integral right-tail area + delta
        area_hist = histogram_right_tail_area(ratios, T, bin_edges)
        delta_area_hist = area_hist - A_hist_baseline

        mean_v = float(np.mean(ratios))
        med_v = float(np.median(ratios))
        std_v = float(np.std(ratios))

        print(
            f"{label:10s} | "
            f"P(focus>T)={p_above:.4f} | "
            f"ΔP={delta_p:+.4f} | "
            f"Area_hist(T..)= {area_hist:.4f} | "
            f"ΔArea_hist={delta_area_hist:+.4f} | "
            f"mean={mean_v:.4f} | "
            f"median={med_v:.4f}"
        )

        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": dataset_name,
            "data_dir": data_dir,
            "img_size": list(img_size),
            "model_label": label,
            "model_path": model_paths.get(label, ""),
            "threshold_source": cfg.get("threshold_source", "baseline_median"),
            "threshold_T": T,

            # CORE METRICS
            "P_focus_above_T": p_above,
            "delta_P_vs_baseline": delta_p,   # <<< KAYITLI

            # (NEW) Histogram-integral metrics (PDF area from T to max)
            "Area_hist_T_to_max": area_hist,
            "delta_Area_hist_vs_baseline": delta_area_hist,
            "hist_bins": int(cfg.get("hist_bins", 50)),
            "hist_x_min": x_min,
            "hist_x_max": x_max,

            "mean_focus": mean_v,
            "median_focus": med_v,
            "std_focus": std_v,

            "N_total": stats_by_model[label]["N_total"],
            "N_face": stats_by_model[label]["N_face"],
            "face_rate": stats_by_model[label]["face_rate"],
            "mask": {
                "roi": "mouth_jaw",
                "roi_padding_px": cfg["roi_padding_px"],
                "background_mask_value": cfg["background_mask_value"],
                "roi_landmark_count": len(ROI_IDX),
            }
        }
        summary_rows.append(summary)

        append_jsonl(summary_path, summary)

    print(f"\n[SAVE] Summary appended to: {summary_path}")
    print_summary_table(summary_rows, dataset_name)
    print_summary_table_markdown(summary_rows, dataset_name)


    # 4) Plot histograms (optional but useful)
    hist_path = os.path.join("artifacts", f"model_focus_comparison_mouth-jaw-{cfg['roi_padding_px']}_{dataset_name}.png")
    plot_focus_ratio_by_model(ratios_by_model, dataset_name, hist_path)
    #plot_histograms(ratios_by_model, dataset_name, hist_path)


    print("\n[DONE]")
