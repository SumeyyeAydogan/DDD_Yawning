"""
Run all CV weight experiments in one process: baseline (no weights) + optimized + log + exp.

Usage (from repo root):
  python scripts/cv_pipeline.py
  python scripts/cv_pipeline.py --cv_epochs 10 --cv_folds 5
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.cross_validation import cross_validate_model


def _write_cv_summary(
    run_dir: str,
    filename: str,
    cv_base_dir: str,
    cv_folds: int,
    cv_epochs: int,
    weights_path: Optional[str],
    cv_results: Dict[str, Any],
) -> None:
    path = os.path.join(run_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("Cross-validation summary\n")
        f.write(f"Experiment    : {filename}\n")
        f.write(f"Folds         : {cv_folds}\n")
        f.write(f"Epochs/fold   : {cv_epochs}\n")
        f.write(f"Base dir      : {cv_base_dir}\n\n")
        f.write(f"Weights file  : {weights_path or 'None'}\n\n")
        f.write(
            f"val_accuracy_mean = {cv_results.get('val_accuracy_mean', float('nan')):.4f}\n"
        )
        f.write(
            f"val_accuracy_std  = {cv_results.get('val_accuracy_std', float('nan')):.4f}\n"
        )
        f.write(f"val_auc_mean      = {cv_results.get('val_auc_mean', float('nan')):.4f}\n")
        f.write(f"val_auc_std       = {cv_results.get('val_auc_std', float('nan')):.4f}\n")
        f.write(f"weight_tag        = {cv_results.get('weight_tag', '')}\n")
        f.write(f"cv_models_dir     = {cv_results.get('cv_models_dir', 'None')}\n")
        f.write(f"cv_logs_dir       = {cv_results.get('cv_logs_dir', 'None')}\n")
        f.write(f"cv_fold_metrics   = {cv_results.get('cv_fold_metrics_path', 'None')}\n")
    print(f"[CV pipeline] Summary saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all CV weight experiments in one run.")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--cv_epochs", type=int, default=30)
    parser.add_argument(
        "--cv_base_dir",
        type=str,
        default="ydd_splitted_dataset_cv",
        help="Dataset root containing train/ (relative to repo root)",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default="artifacts/reward-landmark-soft_cv-roi10",
        help="Folder with exp_weights.json, log_weights.json, optimized_gradcam_weights.json",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="cv_all_weight_experiments",
        help="RunManager run name under runs/",
    )
    args = parser.parse_args()

    project_root = str(_REPO_ROOT)

    cv_base_dir = os.path.join(project_root, args.cv_base_dir)
    if not os.path.isdir(os.path.join(cv_base_dir, "train")):
        raise FileNotFoundError(
            f"CV train folder not found: {os.path.join(cv_base_dir, 'train')}"
        )

    artifacts = os.path.join(project_root, args.artifacts_dir)

    experiments: List[Tuple[str, Optional[str], str]] = [
        ("baseline", None, "cv_summary_baseline.txt"),
        (
            "optimized",
            os.path.join(artifacts, "optimized_gradcam_weights.json"),
            "cv_summary_optimized.txt",
        ),
        ("log", os.path.join(artifacts, "log_weights.json"), "cv_summary_log.txt"),
        ("exp", os.path.join(artifacts, "exp_weights.json"), "cv_summary_exp.txt"),
    ]

    from src.run_manager import RunManager

    print("Starting CV pipeline (all experiments share one run folder)...")
    print("=" * 50)
    run_manager = RunManager(args.run_name)
    print(f"Run directory: {run_manager.run_dir}")
    print(tf.__version__)
    print(tf.config.list_physical_devices("GPU"))

    for label, weights_rel, summary_name in experiments:
        wpath: Optional[str] = weights_rel
        if wpath is not None:
            wpath = os.path.normpath(wpath)
            if not os.path.isfile(wpath):
                print(f"[WARN] Skip '{label}': weights file missing: {wpath}")
                continue

        print(f"\n===== CV experiment: {label} =====")
        print(f"CV data: {cv_base_dir}")
        print(f"Weights: {wpath or 'None'}")

        cv_results = cross_validate_model(
            base_dir=cv_base_dir,
            k=args.cv_folds,
            img_size=(224, 224),
            batch_size=32,
            epochs=args.cv_epochs,
            class_names=("NoYawn", "Yawn"),
            sample_weights_path=wpath,
            run_dir=run_manager.run_dir,
        )
        print(f"CV results ({label}): {cv_results}")

        _write_cv_summary(
            run_manager.run_dir,
            summary_name,
            cv_base_dir,
            args.cv_folds,
            args.cv_epochs,
            wpath,
            cv_results,
        )

    print("\n" + "=" * 50)
    print(f"All finished. Artifacts under: {run_manager.run_dir}")
