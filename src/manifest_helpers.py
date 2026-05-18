from __future__ import annotations

from typing import List, Tuple

import numpy as np
import tensorflow as tf


def read_manifest(manifest_path: str) -> List[str]:
    files: List[str] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                files.append(p)
    return files


def infer_binary_label_from_path(file_path: str, class_names: List[str]) -> float:
    if len(class_names) != 2:
        raise ValueError(f"Expected binary class_names of length 2, got {class_names}")
    norm = file_path.replace("\\", "/")
    if f"/{class_names[0]}/" in norm:
        return 0.0
    if f"/{class_names[1]}/" in norm:
        return 1.0
    raise ValueError(f"Could not infer label from path: {file_path}")


def make_dataset_from_manifest(
    manifest_path: str,
    class_names: List[str],
    img_size: Tuple[int, int],
) -> tf.data.Dataset:
    file_paths = read_manifest(manifest_path)
    if len(file_paths) == 0:
        raise ValueError(f"Manifest has no files: {manifest_path}")

    labels = np.array(
        [infer_binary_label_from_path(fp, class_names) for fp in file_paths],
        dtype=np.float32,
    )
    path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)

    def _load(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, tf.expand_dims(label, axis=-1), path

    ds = tf.data.Dataset.zip((path_ds, label_ds))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(1).prefetch(tf.data.AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds
