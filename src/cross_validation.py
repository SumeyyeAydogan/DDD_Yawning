import os
import json
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from .model import build_model
from .train import train_model
from .utils import plot_history, plot_metrics
from .evaluate import evaluate_model


def _load_full_train_paths_and_labels(
    base_dir: str,
    img_size: Tuple[int, int],
    class_names: Tuple[str, str],
) -> Tuple[List[str], np.ndarray]:
    """
    Load all image file paths and labels from base_dir using Keras utility
    (batch_size=1, shuffle=False so that file_paths aligns with labels).
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        base_dir,
        labels="inferred",
        label_mode="binary",
        class_names=list(class_names),
        image_size=img_size,
        batch_size=1,
        shuffle=False,
    )

    file_paths = list(getattr(ds, "file_paths", []))

    labels_list: List[float] = []
    for _, y in ds:
        # y shape: (1, 1) for binary label_mode
        labels_list.append(float(y.numpy()[0][0]))

    labels = np.array(labels_list, dtype=np.float32)
    return file_paths, labels


def _make_tf_dataset_from_paths(
    file_paths: List[str],
    labels: np.ndarray,
    img_size: Tuple[int, int],
    batch_size: int,
    augment: bool,
    sample_weights: np.ndarray = None,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from lists of paths and labels.
    Uses simple augmentation for train and only rescale for val.
    """
    AUTOTUNE = tf.data.AUTOTUNE

    paths_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)

    if sample_weights is not None:
        weights_ds = tf.data.Dataset.from_tensor_slices(sample_weights.astype(np.float32))
        ds = tf.data.Dataset.zip((paths_ds, labels_ds, weights_ds))
    else:
        ds = tf.data.Dataset.zip((paths_ds, labels_ds))

    def _load_and_preprocess_with_weights(path, label, weight):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, tf.expand_dims(label, axis=-1), tf.cast(weight, tf.float32)

    def _load_and_preprocess(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, tf.expand_dims(label, axis=-1)

    if sample_weights is not None:
        ds = ds.map(_load_and_preprocess_with_weights, num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)

    if augment:
        aug = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomTranslation(0.1, 0.1),
            ]
        )

        if sample_weights is not None:
            def _apply_augment_with_weights(x, y, w):
                return aug(x, training=True), y, w

            ds = ds.map(_apply_augment_with_weights, num_parallel_calls=AUTOTUNE)
        else:
            def _apply_augment(x, y):
                return aug(x, training=True), y

            ds = ds.map(_apply_augment, num_parallel_calls=AUTOTUNE)

    if augment:
        ds = ds.shuffle(1000, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def cross_validate_model(
    base_dir: str,
    k: int = 5,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    seed: int = 42,
    epochs: int = 30,
    class_names: Tuple[str, str] = ("NoYawn", "Yawn"),
    sample_weights_path: str = None,
    fold_sample_weights_dir: str = None,
    run_dir: str = None,
) -> Dict[str, float]:
    """
    Simple k-fold cross-validation over base_dir.

    - Uses only the existing train split for CV.
    - Test split stays untouched for final evaluation later.
    - For each fold: build a fresh model, train on (k-1)/k of train data,
      validate on remaining 1/k, collect val_accuracy and val_auc.
    """
    file_paths, labels = _load_full_train_paths_and_labels(
        base_dir=base_dir,
        img_size=img_size,
        class_names=class_names,
    )

    n_samples = len(file_paths)
    if n_samples == 0:
        raise ValueError(f"No training images found under {base_dir}")

    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)

    # Optional global precomputed sample weights (rel_path -> weight)
    global_weights_by_path = None
    if sample_weights_path:
        if not os.path.exists(sample_weights_path):
            raise FileNotFoundError(f"Sample weights file not found: {sample_weights_path}")
        with open(sample_weights_path, "r", encoding="utf-8") as f:
            global_weights_by_path = json.load(f)
        print(f"[CV] Loaded global sample weights from: {sample_weights_path}")
    if fold_sample_weights_dir:
        if not os.path.isdir(fold_sample_weights_dir):
            raise FileNotFoundError(
                f"Fold sample weights directory not found: {fold_sample_weights_dir}"
            )
        print(f"[CV] Using fold-specific sample weights from: {fold_sample_weights_dir}")

    val_acc_per_fold: List[float] = []
    val_auc_per_fold: List[float] = []

    # Always save fold artifacts when run_dir is provided.
    fold_models_dir = None
    fold_logs_dir = None
    fold_metrics_path = None
    weight_tag = "no_weights"
    if fold_sample_weights_dir:
        weight_tag = os.path.basename(os.path.normpath(fold_sample_weights_dir))
    elif sample_weights_path:
        weight_tag = os.path.splitext(os.path.basename(sample_weights_path))[0]
    if run_dir:
        fold_models_dir = os.path.join(run_dir, "cv_models", weight_tag)
        fold_logs_dir = os.path.join(run_dir, "cv_logs", weight_tag)
        fold_logs_metrics_dir = os.path.join(fold_logs_dir, "metrics")
        fold_logs_plots_dir = os.path.join(fold_logs_dir, "plots")
        fold_logs_manifests_dir = os.path.join(fold_logs_dir, "manifests")
        os.makedirs(fold_models_dir, exist_ok=True)
        os.makedirs(fold_logs_metrics_dir, exist_ok=True)
        os.makedirs(fold_logs_plots_dir, exist_ok=True)
        os.makedirs(fold_logs_manifests_dir, exist_ok=True)
        fold_metrics_path = os.path.join(fold_logs_metrics_dir, "cv_fold_metrics.jsonl")
        print(f"[CV] Fold models will be saved to: {fold_models_dir}")
        print(f"[CV] Fold logs will be saved to: {fold_logs_dir}")

    for fold_idx in range(k):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != fold_idx])

        train_files = [file_paths[i] for i in train_idx]
        val_files = [file_paths[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        train_sample_weights = None
        fold_weights_by_path = None
        if fold_sample_weights_dir:
            fold_id = fold_idx + 1
            fold_weights_path = os.path.join(
                fold_sample_weights_dir, f"fold_{fold_id}_weights.json"
            )
            if not os.path.isfile(fold_weights_path):
                raise FileNotFoundError(
                    f"Fold weights JSON missing for fold {fold_id}: {fold_weights_path}"
                )
            with open(fold_weights_path, "r", encoding="utf-8") as f:
                fold_weights_by_path = json.load(f)

        active_weights = fold_weights_by_path if fold_weights_by_path is not None else global_weights_by_path
        if active_weights is not None:
            train_sample_weights = np.array(
                [
                    float(active_weights.get(os.path.relpath(fp, base_dir).replace("\\", "/"), 1.0))
                    for fp in train_files
                ],
                dtype=np.float32,
            )

        train_ds = _make_tf_dataset_from_paths(
            train_files,
            train_labels,
            img_size,
            batch_size,
            augment=True,
            sample_weights=train_sample_weights,
        )
        val_ds = _make_tf_dataset_from_paths(
            val_files, val_labels, img_size, batch_size, augment=False
        )

        model = build_model(input_shape=(img_size[0], img_size[1], 3))

        print(f"\n===== Fold {fold_idx + 1}/{k} =====")
        history = train_model(
            model,
            train_ds,
            val_ds,
            epochs=epochs,
            callbacks=None,
            initial_epoch=0,
        )

        # Use best validation metrics across epochs.
        val_acc = float(np.nanmax(history.history.get("val_accuracy", [float("nan")])))
        val_auc = float(np.nanmax(history.history.get("val_auc", [float("nan")])))

        print(
            f"Fold {fold_idx + 1}: val_accuracy={val_acc:.4f} | val_auc={val_auc:.4f}"
        )

        if val_acc is not None:
            val_acc_per_fold.append(float(val_acc))
        if val_auc is not None:
            val_auc_per_fold.append(float(val_auc))

        # Save fold model + train/val file lists for follow-up focus analysis
        if run_dir:
            fold_id = fold_idx + 1
            model_path = os.path.join(fold_models_dir, f"fold_{fold_id}.h5")
            model.save(model_path)

            # Save fold training history + plots for baseline/weighted CV visibility
            fold_history_path = os.path.join(fold_logs_metrics_dir, f"fold_{fold_id}_history.json")
            with open(fold_history_path, "w", encoding="utf-8") as hf:
                json.dump(history.history, hf, indent=2)
            fold_history_plot_path = os.path.join(fold_logs_plots_dir, f"fold_{fold_id}_history.png")
            fold_metrics_plot_path = os.path.join(fold_logs_plots_dir, f"fold_{fold_id}_metrics.png")
            plot_history(history, save_path=fold_history_plot_path)
            plot_metrics(history, save_path=fold_metrics_plot_path)

            fold_eval_plots_dir = os.path.join(run_dir, "plots", f"cv_fold_{fold_id}_val")
            os.makedirs(fold_eval_plots_dir, exist_ok=True)
            evaluate_model(
                model,
                val_ds,
                plots_dir=fold_eval_plots_dir,
                class_names=list(class_names),
                subject_diverse_dir=None,
                ds_name="val",
            )

            train_manifest_path = os.path.join(fold_logs_manifests_dir, f"fold_{fold_id}_train_files.txt")
            with open(train_manifest_path, "w", encoding="utf-8") as mf:
                for p in train_files:
                    mf.write(f"{p}\n")

            val_manifest_path = os.path.join(fold_logs_manifests_dir, f"fold_{fold_id}_val_files.txt")
            with open(val_manifest_path, "w", encoding="utf-8") as mf:
                for p in val_files:
                    mf.write(f"{p}\n")

            metric_row = {
                "fold": fold_id,
                "weight_tag": weight_tag,
                "model_path": model_path,
                "train_manifest_path": train_manifest_path,
                "val_manifest_path": val_manifest_path,
                "history_path": fold_history_path,
                "history_plot_path": fold_history_plot_path,
                "metrics_plot_path": fold_metrics_plot_path,
                "eval_plots_dir": fold_eval_plots_dir,
                "subject_diverse_dir": None,
                "train_size": int(len(train_files)),
                "val_size": int(len(val_files)),
                "val_accuracy": float(val_acc) if val_acc is not None else None,
                "val_auc": float(val_auc) if val_auc is not None else None,
            }
            with open(fold_metrics_path, "a", encoding="utf-8") as ff:
                ff.write(json.dumps(metric_row, ensure_ascii=False) + "\n")

    acc_mean = float(np.mean(val_acc_per_fold)) if val_acc_per_fold else float("nan")
    acc_std = float(np.std(val_acc_per_fold)) if val_acc_per_fold else float("nan")
    auc_mean = float(np.mean(val_auc_per_fold)) if val_auc_per_fold else float("nan")
    auc_std = float(np.std(val_auc_per_fold)) if val_auc_per_fold else float("nan")

    print("\n===== Cross-validation summary =====")
    print(f"val_accuracy: mean={acc_mean:.4f}, std={acc_std:.4f}")
    print(f"val_auc     : mean={auc_mean:.4f}, std={auc_std:.4f}")

    results = {
        "val_accuracy_mean": acc_mean,
        "val_accuracy_std": acc_std,
        "val_auc_mean": auc_mean,
        "val_auc_std": auc_std,
        "weight_tag": weight_tag,
    }
    if run_dir:
        results["cv_models_dir"] = fold_models_dir
        results["cv_logs_dir"] = fold_logs_dir
        results["cv_fold_metrics_path"] = fold_metrics_path
    return results