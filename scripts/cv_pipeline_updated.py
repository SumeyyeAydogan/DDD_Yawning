"""
CV pipeline (simplified):
1) Run baseline k-fold CV once (no sample weights).
2) For each fold: build GradCAM weights from that fold's model + train manifest.
3) For each fold & each weight type: run plain (non-CV) training on that fold's train split.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.cross_validation import cross_validate_model, _make_tf_dataset_from_paths
from src.evaluate import evaluate_model
from src.manifest_helpers import infer_binary_label_from_path, read_manifest
from src.model import build_model
from src.train import train_model
from src.utils import plot_history, plot_metrics
from scripts.gradcam_weights_pipeline import (
    build_gradcam_weights_from_train_manifest as build_weights_from_manifest,
)


def _validate_file(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def _validate_baseline_row(row: Dict[str, Any]) -> None:
    required_keys = ["fold", "model_path", "train_manifest_path", "val_manifest_path"]
    missing = [k for k in required_keys if k not in row]
    if missing:
        raise KeyError(f"Baseline fold row is missing keys {missing}: {row}")
    _validate_file(str(row["model_path"]), "baseline fold model")
    _validate_file(str(row["train_manifest_path"]), "baseline train manifest")
    _validate_file(str(row["val_manifest_path"]), "baseline val manifest")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_class_names(raw: str) -> Tuple[str, str]:
    parts = [x.strip() for x in str(raw).split(",") if x.strip()]
    if len(parts) != 2:
        raise SystemExit("--class_names must be two comma-separated names, e.g. NoYawn,Yawn")
    return (parts[0], parts[1])


def train_one_fold_plain_with_weights(
    *,
    train_manifest_path: str,
    val_manifest_path: str,
    train_root: str,
    weights_json_path: str,
    class_names: Tuple[str, str],
    img_size: Tuple[int, int],
    batch_size: int,
    epochs: int,
    model_save_path: str,
    model_plots_dir: str,
    eval_plots_dir: str,
) -> Dict[str, Any]:
    """
    Train a single model on one fold's train subset (no CV),
    using `weights_json_path` as per-image `sample_weight`.
    """
    train_files = read_manifest(train_manifest_path)
    val_files = read_manifest(val_manifest_path)

    y_train = np.array(
        [infer_binary_label_from_path(fp, list(class_names)) for fp in train_files],
        dtype=np.float32,
    )
    y_val = np.array(
        [infer_binary_label_from_path(fp, list(class_names)) for fp in val_files],
        dtype=np.float32,
    )

    with open(weights_json_path, "r", encoding="utf-8") as f:
        weights_map = json.load(f)

    sample_weights = np.array(
        [
            float(
                weights_map.get(
                    os.path.relpath(fp, train_root).replace("\\", "/"),
                    1.0,
                )
            )
            for fp in train_files
        ],
        dtype=np.float32,
    )

    train_ds = _make_tf_dataset_from_paths(
        train_files,
        y_train,
        img_size,
        batch_size,
        augment=True,
        sample_weights=sample_weights,
    )
    val_ds = _make_tf_dataset_from_paths(
        val_files,
        y_val,
        img_size,
        batch_size,
        augment=False,
        sample_weights=None,
    )

    tf.keras.backend.clear_session()
    model = build_model(input_shape=(img_size[0], img_size[1], 3))
    history = train_model(
        model,
        train_ds,
        val_ds,
        epochs=epochs,
        callbacks=None,
        initial_epoch=0,
    )

    val_accuracy = float(np.nanmax(history.history.get("val_accuracy", [float("nan")])))
    val_auc = float(np.nanmax(history.history.get("val_auc", [float("nan")])))

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)

    os.makedirs(model_plots_dir, exist_ok=True)
    model_stem = os.path.splitext(os.path.basename(model_save_path))[0]
    history_plot_path = os.path.join(model_plots_dir, f"{model_stem}_history.png")
    metrics_plot_path = os.path.join(model_plots_dir, f"{model_stem}_metrics.png")
    plot_history(history, save_path=history_plot_path)
    plot_metrics(history, save_path=metrics_plot_path)

    os.makedirs(eval_plots_dir, exist_ok=True)
    evaluate_model(
        model,
        val_ds,
        plots_dir=eval_plots_dir,
        class_names=list(class_names),
        subject_diverse_dir=None,
        ds_name="val",
    )

    return {
        "val_accuracy": float(val_accuracy) if val_accuracy is not None else float("nan"),
        "val_auc": float(val_auc) if val_auc is not None else float("nan"),
        "history_plot_path": history_plot_path,
        "metrics_plot_path": metrics_plot_path,
        "eval_plots_dir": eval_plots_dir,
    }


def _write_baseline_summary(
    *,
    run_dir: str,
    baseline_results: Dict[str, float],
    baseline_fold_metrics_path: str,
) -> str:
    summary_path = os.path.join(run_dir, "baseline_cv_summary.txt")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Baseline CV summary\n")
        f.write(f"cv_fold_metrics_path: {baseline_fold_metrics_path}\n\n")
        f.write(f"val_accuracy_mean: {baseline_results.get('val_accuracy_mean', float('nan'))}\n")
        f.write(f"val_accuracy_std : {baseline_results.get('val_accuracy_std', float('nan'))}\n")
        f.write(f"val_auc_mean     : {baseline_results.get('val_auc_mean', float('nan'))}\n")
        f.write(f"val_auc_std      : {baseline_results.get('val_auc_std', float('nan'))}\n")
    return summary_path


def _append_plain_training_summary(
    *,
    summary_path: str,
    fold_id: int,
    weight_name: str,
    weights_json_path: str,
    model_save_path: str,
    eval_plots_dir: str,
    val_accuracy: float,
    val_auc: float,
) -> None:
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    file_exists = os.path.isfile(summary_path)
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "fold_id",
                    "weight_type",
                    "weights_json_path",
                    "model_path",
                    "eval_plots_dir",
                    "val_accuracy",
                    "val_auc",
                ]
            )
        writer.writerow(
            [
                fold_id,
                weight_name,
                weights_json_path,
                model_save_path,
                eval_plots_dir,
                val_accuracy,
                val_auc,
            ]
        )

def _write_final_plain_summary(summary_csv_path: str) -> str:
    grouped: Dict[str, Dict[str, List[float]]] = {}

    with open(summary_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            weight_type = row["weight_type"]
            grouped.setdefault(weight_type, {"val_accuracy": [], "val_auc": []})
            grouped[weight_type]["val_accuracy"].append(float(row["val_accuracy"]))
            grouped[weight_type]["val_auc"].append(float(row["val_auc"]))

    out_path = os.path.splitext(summary_csv_path)[0] + ".txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for weight_type in ["optimized", "exp", "log"]:
            if weight_type not in grouped:
                continue

            acc = np.array(grouped[weight_type]["val_accuracy"], dtype=np.float32)
            auc = np.array(grouped[weight_type]["val_auc"], dtype=np.float32)

            f.write(f"{weight_type}\n")
            f.write(f"val_accuracy_mean: {acc.mean()}\n")
            f.write(f"val_accuracy_std : {acc.std()}\n")
            f.write(f"val_auc_mean     : {auc.mean()}\n")
            f.write(f"val_auc_std      : {auc.std()}\n\n")

    return out_path


def _summarize_from_fold_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    acc_vals: List[float] = []
    auc_vals: List[float] = []
    for row in rows:
        try:
            acc_vals.append(float(row.get("val_accuracy", float("nan"))))
        except Exception:
            pass
        try:
            auc_vals.append(float(row.get("val_auc", float("nan"))))
        except Exception:
            pass
    acc = np.array(acc_vals, dtype=np.float32) if acc_vals else np.array([np.nan], dtype=np.float32)
    auc = np.array(auc_vals, dtype=np.float32) if auc_vals else np.array([np.nan], dtype=np.float32)
    return {
        "val_accuracy_mean": float(np.nanmean(acc)),
        "val_accuracy_std": float(np.nanstd(acc)),
        "val_auc_mean": float(np.nanmean(auc)),
        "val_auc_std": float(np.nanstd(auc)),
    }


def _load_completed_fold_weight_pairs(summary_csv_path: str) -> set[Tuple[int, str]]:
    completed: set[Tuple[int, str]] = set()
    if not os.path.isfile(summary_csv_path):
        return completed
    with open(summary_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fold_id = int(row.get("fold_id", -1))
            except Exception:
                continue
            weight_type = str(row.get("weight_type", "")).strip().lower()
            if fold_id >= 1 and weight_type:
                completed.add((fold_id, weight_type))
    return completed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline CV -> fold weights -> plain fold training.")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--cv_epochs", type=int, default=30, help="Epochs per fold (baseline CV + plain training)")
    parser.add_argument(
        "--cv_base_dir",
        type=str,
        default="ydd_splitted_dataset_cv",
        help="Dataset root",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default="artifacts/gradcam_fold_weights_cvflow",
        help="Where fold-wise weights are written",
    )
    parser.add_argument("--img_size", type=str, default="224,224", help="H,W")
    parser.add_argument(
        "--class_names",
        type=str,
        default="NoYawn,Yawn",
        help="Binary class folder names in order class0,class1 (e.g. NoYawn,Yawn)",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--search_level", type=int, default=2)
    parser.add_argument("--weight_mode", type=str, default="reward", choices=["reward", "penalize"])
    parser.add_argument("--roi_padding_px", type=int, default=6)
    parser.add_argument("--background_mask_value", type=float, default=0.2)
    parser.add_argument("--fallback_to_static", type=int, default=1, choices=[0, 1])
    parser.add_argument("--run_name", type=str, default="cv_baseline_then_fold_train")
    parser.add_argument(
        "--resume_run_name",
        type=str,
        default="",
        help="Resume from an existing run directory under runs/<name>.",
    )
    parser.add_argument(
        "--skip_baseline_if_exists",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, reuse existing baseline fold metrics when available.",
    )
    parser.add_argument(
        "--baseline_only",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, runs only baseline CV and exits.",
    )
    parser.add_argument(
        "--continue_on_fold_error",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, continue other folds when one fold fails in weighting/plain training.",
    )
    args = parser.parse_args()

    project_root = str(_REPO_ROOT)
    cv_base_dir = os.path.join(project_root, args.cv_base_dir)
    if not os.path.isdir(cv_base_dir):
        raise FileNotFoundError(f"dataset root not found: {cv_base_dir}")

    h_w = [int(x.strip()) for x in str(args.img_size).split(",") if x.strip()]
    if len(h_w) != 2:
        raise SystemExit("--img_size must be like 224,224")
    img_size = (h_w[0], h_w[1])
    class_names: Tuple[str, str] = _parse_class_names(args.class_names)

    from src.run_manager import RunManager

    auto_opt_script = os.path.join(project_root, "scripts", "auto_optimize_gradcam_weights.py")
    log_exp_script = os.path.join(project_root, "scripts", "log_exp_script.py")
    _validate_file(auto_opt_script, "GradCAM auto-opt script")
    _validate_file(log_exp_script, "GradCAM log/exp conversion script")

    print("Starting CV pipeline (simplified)...")
    resolved_run_name = args.resume_run_name.strip() if str(args.resume_run_name).strip() else args.run_name
    run_manager = RunManager(resolved_run_name)
    print(f"Run directory: {run_manager.run_dir}")
    print(f"Class names: {class_names}")

    # 1) Baseline CV (no weights)
    existing_baseline_fold_metrics_path = os.path.join(
        run_manager.run_dir, "cv_logs", "no_weights", "metrics", "cv_fold_metrics.jsonl"
    )
    if int(args.skip_baseline_if_exists) == 1 and os.path.isfile(existing_baseline_fold_metrics_path):
        print(f"[Resume] Reusing baseline fold metrics: {existing_baseline_fold_metrics_path}")
        baseline_fold_metrics_path = existing_baseline_fold_metrics_path
        baseline_rows_for_summary = _read_jsonl(baseline_fold_metrics_path)
        baseline_results = _summarize_from_fold_rows(baseline_rows_for_summary)
        baseline_results["cv_fold_metrics_path"] = baseline_fold_metrics_path
    else:
        baseline_results = cross_validate_model(
            base_dir=cv_base_dir,
            k=args.cv_folds,
            img_size=img_size,
            batch_size=args.batch_size,
            epochs=args.cv_epochs,
            class_names=class_names,
            sample_weights_path=None,
            fold_sample_weights_dir=None,
            run_dir=run_manager.run_dir,
        )
        baseline_fold_metrics_path = baseline_results.get("cv_fold_metrics_path")
        if not baseline_fold_metrics_path or not os.path.isfile(baseline_fold_metrics_path):
            raise RuntimeError("Baseline fold metrics are missing; cannot build fold-wise weights.")

    _baseline_summary_path = _write_baseline_summary(
        run_dir=run_manager.run_dir,
        baseline_results=baseline_results,
        baseline_fold_metrics_path=baseline_fold_metrics_path,
    )
    plain_training_summary_path = os.path.join(run_manager.run_dir, "plain_training_summary.csv")
    print(f"[Summary] Baseline: {_baseline_summary_path}")
    if int(args.baseline_only) == 1:
        print("[Done] Baseline-only mode enabled; skipping weight generation and plain training.")
        raise SystemExit(0)

    baseline_rows = _read_jsonl(baseline_fold_metrics_path)
    if len(baseline_rows) == 0:
        raise RuntimeError(f"No rows found in baseline fold metrics: {baseline_fold_metrics_path}")

    # 2) Build fold-wise weights + 3) Plain training for each weight type
    artifacts = os.path.join(project_root, args.artifacts_dir)
    optimized_dir = os.path.join(artifacts, "fold_weights_optimized")
    log_dir = os.path.join(artifacts, "fold_weights_log")
    exp_dir = os.path.join(artifacts, "fold_weights_exp")
    tmp_work_dir = os.path.join(artifacts, "_tmp_auto_opt")

    non_cv_root = os.path.join(run_manager.run_dir, "non_cv_models")
    non_cv_models_root = os.path.join(non_cv_root, "models")
    non_cv_plots_root = os.path.join(non_cv_root, "plots")
    weight_types: List[Tuple[str, str]] = [
        ("optimized", "optimized_path"),
        ("log", "log_path"),
        ("exp", "exp_path"),
    ]

    failed_folds: List[int] = []
    completed_pairs = _load_completed_fold_weight_pairs(plain_training_summary_path)
    if completed_pairs:
        print(f"[Resume] Found {len(completed_pairs)} completed fold/weight runs in summary CSV.")
    for row in baseline_rows:
        _validate_baseline_row(row)
        fold_id = int(row["fold"])
        model_path = str(row["model_path"])
        train_manifest_path = str(row["train_manifest_path"])
        val_manifest_path = str(row["val_manifest_path"])

        try:
            print(f"\n=== Fold {fold_id}: build weights ===")
            expected_weight_paths = {
                "optimized_path": os.path.join(optimized_dir, f"fold_{fold_id}_weights.json"),
                "log_path": os.path.join(log_dir, f"fold_{fold_id}_weights.json"),
                "exp_path": os.path.join(exp_dir, f"fold_{fold_id}_weights.json"),
            }
            if all(os.path.isfile(p) for p in expected_weight_paths.values()):
                print(f"[Resume] Fold {fold_id}: reusing existing optimized/log/exp weight JSONs.")
                weight_paths = expected_weight_paths
            else:
                weight_paths = build_weights_from_manifest(
                    model_path=model_path,
                    train_root=cv_base_dir,
                    train_manifest_path=train_manifest_path,
                    optimized_dir=optimized_dir,
                    log_dir=log_dir,
                    exp_dir=exp_dir,
                    tmp_work_dir=tmp_work_dir,
                    img_size=f"{img_size[0]},{img_size[1]}",
                    search_level=args.search_level,
                    weight_mode=args.weight_mode,
                    roi_padding_px=args.roi_padding_px,
                    background_mask_value=args.background_mask_value,
                    fallback_to_static=args.fallback_to_static,
                )

            for weight_name, key in weight_types:
                if (fold_id, weight_name.lower()) in completed_pairs:
                    print(f"[Resume] Fold {fold_id} ({weight_name}) already completed; skipping.")
                    continue
                weights_json_path = weight_paths[key]
                _validate_file(weights_json_path, f"{weight_name} weights json")
                model_dir = os.path.join(non_cv_models_root, weight_name)
                plots_dir = os.path.join(non_cv_plots_root, weight_name)
                os.makedirs(model_dir, exist_ok=True)
                os.makedirs(plots_dir, exist_ok=True)
                model_save_path = os.path.join(model_dir, f"fold_{fold_id}.h5")

                print(f"=== Fold {fold_id}: plain train ({weight_name}) ===")
                eval_plots_dir = os.path.join(
                    plots_dir, f"fold_{fold_id}_eval_val"
                )
                metrics = train_one_fold_plain_with_weights(
                    train_manifest_path=train_manifest_path,
                    val_manifest_path=val_manifest_path,
                    train_root=cv_base_dir,
                    weights_json_path=weights_json_path,
                    class_names=class_names,
                    img_size=img_size,
                    batch_size=args.batch_size,
                    epochs=args.cv_epochs,
                    model_save_path=model_save_path,
                    model_plots_dir=plots_dir,
                    eval_plots_dir=eval_plots_dir,
                )
                _append_plain_training_summary(
                    summary_path=plain_training_summary_path,
                    fold_id=fold_id,
                    weight_name=weight_name,
                    weights_json_path=weights_json_path,
                    model_save_path=model_save_path,
                    eval_plots_dir=metrics["eval_plots_dir"],
                    val_accuracy=metrics["val_accuracy"],
                    val_auc=metrics["val_auc"],
                )
        except Exception as exc:
            msg = f"[Fold {fold_id}] failed: {exc}"
            if int(args.continue_on_fold_error) == 1:
                print(f"{msg} (continue_on_fold_error=1, skipping fold)")
                failed_folds.append(fold_id)
                continue
            raise RuntimeError(msg) from exc

    final_summary_path = _write_final_plain_summary(plain_training_summary_path)

    print("\nAll finished.")
    print(f"Run directory: {run_manager.run_dir}")
    print(f"[Summary] Plain training final summary: {final_summary_path}")
    if failed_folds:
        print(f"[Summary] Failed folds skipped: {failed_folds}")
