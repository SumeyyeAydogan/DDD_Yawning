import json
import math
import os

INPUT_PATH = "artifacts/reward-landmark-soft-mj-9/optimized_gradcam_weights.json" #"artifacts/reward_soft-mask/optimized_gradcam_weights.json"
LOG_OUTPUT_PATH = "artifacts/reward-landmark-soft-mj-9/log_weights.json"
EXP_OUTPUT_PATH = "artifacts/reward-landmark-soft-mj-9/exp_weights.json"


def create_log_weights(input_path=INPUT_PATH, output_path=LOG_OUTPUT_PATH):
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


def create_exp_weights(input_path=INPUT_PATH, output_path=EXP_OUTPUT_PATH):
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
    create_log_weights()
    create_exp_weights()
