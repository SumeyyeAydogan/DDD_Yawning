"""
End-to-end helper to remove manual steps:

1) Run auto_optimize_gradcam_weights.py to produce optimized_gradcam_weights.json
2) Generate log_weights.json and exp_weights.json from that output

This script only orchestrates files/paths; the weighting logic lives in the existing scripts.
"""

import argparse
import os
import subprocess
import sys


def _run(cmd: list[str]) -> None:
    print("[Run]", " ".join(cmd))
    subprocess.check_call(cmd)


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

