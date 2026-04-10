import os
import json
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from .model import build_model
from .train import train_model


def _load_full_train_paths_and_labels(
    base_dir: str,
    img_size: Tuple[int, int],
    class_names: Tuple[str, str],
) -> Tuple[List[str], np.ndarray]:
    """
    Load all image file paths and labels from base_dir/train using Keras utility
    (batch_size=1, shuffle=False so that file_paths aligns with labels).
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(base_dir, "train"),
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
) -> Dict[str, float]:
    """
    Simple k-fold cross-validation over base_dir/train.

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
        raise ValueError(f"No training images found under {os.path.join(base_dir, 'train')}")

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

    val_acc_per_fold: List[float] = []
    val_auc_per_fold: List[float] = []

    for fold_idx in range(k):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != fold_idx])

        train_files = [file_paths[i] for i in train_idx]
        val_files = [file_paths[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        train_sample_weights = None
        if global_weights_by_path is not None:
            train_root = os.path.join(base_dir, "train")
            train_sample_weights = np.array(
                [
                    float(global_weights_by_path.get(os.path.relpath(fp, train_root).replace("\\", "/"), 1.0))
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

        # History keys: 'loss', 'accuracy', 'precision', 'recall', 'auc', 'val_loss', ...
        val_acc = history.history.get("val_accuracy", [None])[-1]
        val_auc = history.history.get("val_auc", [None])[-1]

        print(
            f"Fold {fold_idx + 1}: val_accuracy={val_acc:.4f} | val_auc={val_auc:.4f}"
        )

        if val_acc is not None:
            val_acc_per_fold.append(float(val_acc))
        if val_auc is not None:
            val_auc_per_fold.append(float(val_auc))

    acc_mean = float(np.mean(val_acc_per_fold)) if val_acc_per_fold else float("nan")
    acc_std = float(np.std(val_acc_per_fold)) if val_acc_per_fold else float("nan")
    auc_mean = float(np.mean(val_auc_per_fold)) if val_auc_per_fold else float("nan")
    auc_std = float(np.std(val_auc_per_fold)) if val_auc_per_fold else float("nan")

    print("\n===== Cross-validation summary =====")
    print(f"val_accuracy: mean={acc_mean:.4f}, std={acc_std:.4f}")
    print(f"val_auc     : mean={auc_mean:.4f}, std={auc_std:.4f}")

    return {
        "val_accuracy_mean": acc_mean,
        "val_accuracy_std": acc_std,
        "val_auc_mean": auc_mean,
        "val_auc_std": auc_std,
    }

