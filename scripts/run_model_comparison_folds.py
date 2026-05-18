import os
from pathlib import Path
import sys

# Add root path (project structure assumed same as before)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from scripts.model_comparison_integral import run_multi_fold_comparison


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    # Must match cv_pipeline default --run_name
    run_name = "cv_baseline_then_fold_train-roi6_best-second"
    run_root = project_root / "runs" / run_name

    # Control both fold ids and map size from one place.
    # Example: fold_start=1, fold_count=5 -> folds 1..5
    fold_start = 1
    fold_count = 5
    fold_ids = list(range(fold_start, fold_start + fold_count))

    # Paths aligned with current cv_pipeline outputs:
    # - baseline models: runs/<run_name>/cv_models/no_weights/fold_<id>.h5
    # - weighted models: runs/<run_name>/non_cv_models/models/{optimized|log|exp}/fold_<id>.h5
    # - val manifests : runs/<run_name>/cv_logs/no_weights/manifests/fold_<id>_val_files.txt
    model_map_by_fold = {}
    val_manifest_by_fold = {}
    for fold_id in fold_ids:
        model_map_by_fold[fold_id] = [
            {"label": "original", "model_path": str(run_root / f"cv_models/no_weights/fold_{fold_id}.h5")},
            {"label": "reward", "model_path": str(run_root / f"non_cv_models/models/optimized/fold_{fold_id}.h5")},
            {"label": "log-reward", "model_path": str(run_root / f"non_cv_models/models/log/fold_{fold_id}.h5")},
            {"label": "exp-reward", "model_path": str(run_root / f"non_cv_models/models/exp/fold_{fold_id}.h5")},
        ]
        val_manifest_by_fold[fold_id] = str(run_root / f"cv_logs/no_weights/manifests/fold_{fold_id}_val_files.txt")

    # Optional: map model labels to research weight types for aggregation.
    weight_type_map_by_label = {
        "original": "original",
        "reward": "reward",
        "log-reward": "log",
        "exp-reward": "exp",
    }

    fold_ids = [f for f in fold_ids if f in model_map_by_fold and f in val_manifest_by_fold]
    output_dir = str(project_root / "artifacts" / "model_comparison")
    os.makedirs(output_dir, exist_ok=True)

    result = run_multi_fold_comparison(
        fold_ids=fold_ids,
        val_manifest_by_fold=val_manifest_by_fold,
        model_map_by_fold=model_map_by_fold,
        output_dir=output_dir,
        experiment_id="manual_fold_comparison",
        weight_type_map_by_label=weight_type_map_by_label,
        config_override={
            "dataset_name": "cv_val",
            "class_names": ["NoYawn", "Yawn"],
            "img_size": (224, 224),
        },
    )

    print("\n[RUN RESULT]")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
