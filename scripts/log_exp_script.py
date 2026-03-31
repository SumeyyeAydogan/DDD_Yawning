import json
import math
import os
import argparse

def create_log_weights(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find {input_path}")

    with open(input_path, "r") as f:
        weights = json.load(f)

    log_weights = {}

    for key, value in weights.items():
        safe_value = max(value, 1e-12)  # log(0) önleme
        log_weights[key] = math.log(safe_value)

    # 2) minimum log değeri bul
    min_log = min(log_weights.values())

    # 3) minimumu ekle (negatifleri kaldır)
    if min_log < 0:
        shift = -min_log
        for k in log_weights:
            log_weights[k] += shift    

    with open(output_path, "w") as f:
        json.dump(log_weights, f, indent=2)

    print(f"[OK] log weights saved to: {output_path}")


def create_exp_weights(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find {input_path}")

    with open(input_path, "r") as f:
        weights = json.load(f)

    exp_weights = {}

    for key, value in weights.items():
        # Exponent alırken aşırı büyük değerlere karşı güvenlik
        safe_value = min(value, 50)  # exp(50) ≈ 5e21 → yeterince büyük
        exp_weights[key] = math.exp(safe_value)

    with open(output_path, "w") as f:
        json.dump(exp_weights, f, indent=2)

    print(f"[OK] exp weights saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create log/exp transformed GradCAM sample weights.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to optimized_gradcam_weights.json",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. If set, outputs log_weights.json and exp_weights.json into it.",
    )
    parser.add_argument(
        "--log-out",
        default=None,
        help="Full output path for log weights json (overrides --out-dir).",
    )
    parser.add_argument(
        "--exp-out",
        default=None,
        help="Full output path for exp weights json (overrides --out-dir).",
    )

    args = parser.parse_args()

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        log_out = args.log_out or os.path.join(args.out_dir, "log_weights.json")
        exp_out = args.exp_out or os.path.join(args.out_dir, "exp_weights.json")
    else:
        if not args.log_out or not args.exp_out:
            raise SystemExit("Either provide --out-dir OR both --log-out and --exp-out.")
        log_out = args.log_out
        exp_out = args.exp_out

    create_log_weights(args.input, log_out)
    create_exp_weights(args.input, exp_out)
