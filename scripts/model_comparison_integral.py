import os, sys, json
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Add root path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.gradcam import CustomGradCAM
from src.ds_with_paths_pipeline import get_dataset_with_paths
from src.focus_metrics import compute_focus_ratio, empirical_right_tail_probability
from src.mask_helpers import create_landmark_mask, image_to_float01_rgb, image_to_uint8_rgb


# ================== CONFIG ======================
CONFIG = {
    "data_dir": r"ydd_splitted_dataset/test",
    "img_size": (224, 224),
    "dataset_name": "test",

    # TF dataset class order for yawning:
    # NoYawn = 0, Yawn = 1
    "class_names": ["NoYawn", "Yawn"],

    # Mask params
    "background_mask_value": 0.2,   # 0.0 = hard mask, 0.2 = soft mask
    "roi_padding_px": 6,
    # When bbox is wide, reduce horizontal padding (pad_x) relative to vertical padding (pad_y).
    "roi_keep_aspect_pad_x_min_scale": 0.2,

    # Threshold rule
    "threshold_source": "baseline_median",

    # Plot
    "hist_bins": 50,
    
    #Gradcam Source
    "gradcam_class_source": "model_prediction"
}

# IMPORTANT: "original" must exist for threshold
MODEL_CONFIGS = [
    {"label": "original",   "model_path": r"runs/30_epoch_baseline_e3_yawning/models/final_model.h5"},
    {"label": "reward",     "model_path": r"runs/30_epoch_reward-mouth-jaw-10-landmark/models/final_model.h5"},
    {"label": "log-reward", "model_path": r"runs/30_epoch_log-reward-mouth-jaw-10-landmark/models/final_model.h5"},
    {"label": "exp-reward", "model_path": r"runs/30_epoch_exp-reward-mouth-jaw-9-landmark/models/final_model.h5"},
]

# ================== CORE ======================
def collect_pred_class_focus_ratios_for_model(model, data_dir, img_size):
    """
    Returns:
      focus_ratios_pred_class: np.array (only face_ok==1)
      stats: dict with N_total, N_face, face_rate
    """
    gradcam = CustomGradCAM(model)

    ds, file_paths = get_dataset_with_paths(data_dir, img_size)

    ratios = []
    face_ok = 0
    total = 0

    for idx, (data_batch, path_batch) in enumerate(ds):
        total += 1
        images, labels = data_batch
        image = images[0].numpy()

        image_rgb_uint8 = image_to_uint8_rgb(image)
        image_float01 = image_to_float01_rgb(image)

        # prediction -> class_idx for gradcam
        prob = float(model.predict(image_float01[None, ...], verbose=0)[0][0])
        pred = 1 if prob >= 0.5 else 0

        # GradCAM heatmap (pred class)
        heatmap = gradcam.compute_heatmap(image_float01, class_idx=pred)
        heatmap = tf.image.resize(
            heatmap[..., None],
            img_size,
            method="bilinear",
            antialias=True
        ).numpy()[..., 0]

        mask = create_landmark_mask(image_rgb_uint8, img_size, CONFIG)
        if mask is None:
            pass
        else:
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

    # Optionally pin axis range:
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
        ratios, stats = collect_pred_class_focus_ratios_for_model(model, data_dir, img_size)

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
    P_baseline = empirical_right_tail_probability(baseline_ratios, T)

    summary_rows = []

    for label, ratios in ratios_by_model.items():
        if len(ratios) == 0:
            continue

        p_above = empirical_right_tail_probability(ratios, T)
        delta_p = p_above - P_baseline

        mean_v = float(np.mean(ratios))
        med_v = float(np.median(ratios))
        std_v = float(np.std(ratios))

        print(
            f"{label:10s} | "
            f"P(focus>T)={p_above:.4f} | "
            f"ΔP={delta_p:+.4f} | "
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
            }
        }
        summary_rows.append(summary)

        append_jsonl(summary_path, summary)

    print(f"\n[SAVE] Summary appended to: {summary_path}")
    print_summary_table(summary_rows, dataset_name)
    print_summary_table_markdown(summary_rows, dataset_name)


    # 4) Plot histograms (optional but useful)
    hist_path = os.path.join("artifacts", f"model_focus_comparison_mask_{dataset_name}.png")
    plot_focus_ratio_by_model(ratios_by_model, dataset_name, hist_path)


    print("\n[DONE]")