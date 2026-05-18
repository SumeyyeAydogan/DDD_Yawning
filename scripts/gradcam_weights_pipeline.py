"""
End-to-end helper to remove manual steps:

1) Run auto_optimize_gradcam_weights.py to produce optimized_gradcam_weights.json
2) Generate log_weights.json and exp_weights.json from that output

This script only orchestrates files/paths; the weighting logic lives in the existing scripts.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from src.manifest_helpers import read_manifest


def _run(cmd: list[str]) -> None:
    print("[Run]", " ".join(cmd))
    subprocess.check_call(cmd)


_REPO_ROOT = Path(__file__).resolve().parent.parent


def build_gradcam_weights_from_train_manifest(
    *,
    model_path: str,
    train_root: str,
    train_manifest_path: str,
    optimized_dir: str,
    log_dir: str,
    exp_dir: str,
    tmp_work_dir: str,
    img_size: str = "224,224",
    search_level: int = 2,
    weight_mode: str = "reward",
    roi_padding_px: int = 6,
    background_mask_value: float = 0.2,
    fallback_to_static: int = 1,
) -> Dict[str, str]:
    """
    Build GradCAM sample weights for exactly the subset described by `train_manifest_path`.

    Notes:
    - `auto_optimize_gradcam_weights.py` computes weights for the whole `train_root`.
      Then we filter keys by paths listed in `train_manifest_path`.
    - `train_manifest_path` should contain absolute file paths (as produced by `cross_validation.py`).
    """
    import re

    if not os.path.isfile(train_manifest_path):
        raise FileNotFoundError(f"train_manifest_path not found: {train_manifest_path}")

    os.makedirs(optimized_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(tmp_work_dir, exist_ok=True)

    # Try extracting fold_id from: fold_{fold_id}_train_files.txt
    manifest_base = os.path.basename(train_manifest_path)
    m = re.search(r"fold_(\d+)_train_files", manifest_base)
    fold_id = int(m.group(1)) if m else None
    out_filename = f"fold_{fold_id}_weights.json" if fold_id is not None else f"{os.path.splitext(manifest_base)[0]}_weights.json"

    auto_opt_script = os.path.join(str(_REPO_ROOT), "scripts", "auto_optimize_gradcam_weights.py")
    log_exp_script = os.path.join(str(_REPO_ROOT), "scripts", "log_exp_script.py")

    # 1) Run auto-opt once for this model
    fold_tmp_dir = os.path.join(tmp_work_dir, os.path.splitext(out_filename)[0])
    os.makedirs(fold_tmp_dir, exist_ok=True)

    _run(
        [
            sys.executable,
            auto_opt_script,
            "--model-path",
            model_path,
            "--data-dir",
            train_root,
            "--manifest-path",
            train_manifest_path,
            "--artifacts-dir",
            fold_tmp_dir,
            "--img-size",
            img_size,
            "--search-level",
            str(search_level),
            "--weight-mode",
            weight_mode,
            "--roi-padding-px",
            str(roi_padding_px),
            "--background-mask-value",
            str(background_mask_value),
            "--fallback-to-static",
            str(fallback_to_static),
        ]
    )

    # 2) Keep only weights matching manifest paths
    auto_json_path = os.path.join(fold_tmp_dir, "optimized_gradcam_weights.json")
    if not os.path.isfile(auto_json_path):
        raise FileNotFoundError(f"auto_opt output missing: {auto_json_path}")

    with open(auto_json_path, "r", encoding="utf-8") as f:
        auto_weights = json.load(f)

    train_files = read_manifest(train_manifest_path)
    fold_weights: Dict[str, float] = {}
    for abs_path in train_files:
        rel = os.path.relpath(abs_path, train_root).replace("\\", "/")
        if rel in auto_weights:
            fold_weights[rel] = float(auto_weights[rel])

    optimized_out_path = os.path.join(optimized_dir, out_filename)
    with open(optimized_out_path, "w", encoding="utf-8") as f:
        json.dump(fold_weights, f, indent=2)

    # 3) Create log/exp versions
    # Keep filenames consistent with the rest of the pipeline:
    # log_dir/exp_dir both use the same name as optimized weights.
    log_out_path = os.path.join(log_dir, out_filename)
    exp_out_path = os.path.join(exp_dir, out_filename)

    _run(
        [
            sys.executable,
            log_exp_script,
            "--input",
            optimized_out_path,
            "--log-out",
            log_out_path,
            "--exp-out",
            exp_out_path,
        ]
    )

    return {
        "optimized_path": optimized_out_path,
        "log_path": log_out_path,
        "exp_path": exp_out_path,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="GradCAM weights pipeline (optimize -> log/exp).")
    parser.add_argument("--model-path", default="runs/30_epoch_baseline_e3_yawning/models/final_model.h5")
    parser.add_argument("--data-dir", default="ydd_splitted_dataset/train", help="Training folder (e.g. ydd_splitted_dataset/train)")
    parser.add_argument("--artifacts-dir", default="artifacts/reward-landmark-soft", help="Output folder for weights (e.g. artifacts/reward-landmark-soft)")
    parser.add_argument("--img-size", default="224,224")
    parser.add_argument("--search-level", type=int, default=2)
    parser.add_argument("--weight-mode", choices=["reward", "penalize"], default="reward")
    parser.add_argument("--roi-padding-px", type=int, default=10)
    parser.add_argument("--background-mask-value", type=float, default=0.2)
    parser.add_argument("--fallback-to-static", type=int, default=1)

    args = parser.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    auto_opt_script = os.path.join(repo_root, "scripts", "auto_optimize_gradcam_weights.py")
    log_exp_script = os.path.join(repo_root, "scripts", "log_exp_script.py")

    _run(
        [
            sys.executable,
            auto_opt_script,
            "--model-path",
            args.model_path,
            "--data-dir",
            args.data_dir,
            "--artifacts-dir",
            args.artifacts_dir,
            "--img-size",
            args.img_size,
            "--search-level",
            str(args.search_level),
            "--weight-mode",
            args.weight_mode,
            "--roi-padding-px",
            str(args.roi_padding_px),
            "--background-mask-value",
            str(args.background_mask_value),
            "--fallback-to-static",
            str(args.fallback_to_static),
        ]
    )

    optimized_path = os.path.join(args.artifacts_dir, "optimized_gradcam_weights.json")
    _run(
        [
            sys.executable,
            log_exp_script,
            "--input",
            optimized_path,
            "--out-dir",
            args.artifacts_dir,
        ]
    )

    print("\n[OK] Pipeline complete.")
    print("Outputs:")
    print(f"  - {optimized_path}")
    print(f"  - {os.path.join(args.artifacts_dir, 'log_weights.json')}")
    print(f"  - {os.path.join(args.artifacts_dir, 'exp_weights.json')}")


if __name__ == "__main__":
    main()