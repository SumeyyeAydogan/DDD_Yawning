import json
import os

import numpy as np
import tensorflow as tf
from src.simple_mask import create_simple_mask_generator


def get_binary_pipelines(
    base_dir,
    img_size=(224, 224),
    batch_size=16,
    seed=42,
    class_names=("NoYawn", "Yawn"),
    use_background_aug=False,  # Adversarial background augmentation
    bg_aug_prob=0.4,           # Probability of applying background augmentation
    bg_aug_face_ratio=0.75,    # Face region ratio (center of image)
    gradcam_weights_path=None,
    gradcam_weight_scale=1.0,  # Don't scale - weights already optimized by auto_optimize script 04 for exp
    gradcam_weight_clip=(0.1, 6.9),  # Match optimized clip range from gradcam_opt_params.json 0.76, 1.24 (0.1, 3.4
    #2-3
):
    AUTOTUNE = tf.data.AUTOTUNE

    # 1) Create datasets (binary labels)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{base_dir}/train",
        labels="inferred",
        label_mode="binary",
        class_names=list(class_names),
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    # This order will be used to match GradCAM weights
    train_file_paths = list(getattr(train_ds, "file_paths", []))

    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{base_dir}/val",
        labels="inferred",
        label_mode="binary",
        class_names=list(class_names),
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    val_ds = val_ds.apply(tf.data.experimental.ignore_errors())

    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"{base_dir}/test",
        labels="inferred",
        label_mode="binary",
        class_names=list(class_names),
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )
    test_ds = test_ds.apply(tf.data.experimental.ignore_errors())

    # 2) Augmentation (train only) + 3) Normalization
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ]
    )
    normalization = tf.keras.layers.Rescaling(1.0 / 255)

    # 4) Apply with map
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE,
    )
    val_ds = val_ds.map(
        lambda x, y: (normalization(x), y),
        num_parallel_calls=AUTOTUNE,
    )
    test_ds = test_ds.map(
        lambda x, y: (normalization(x), y),
        num_parallel_calls=AUTOTUNE,
    )

    # 4.5) Adversarial background augmentation (train only)
    if use_background_aug:
        from src.adversarial_augmentation import RandomBackgroundReplacement

        bg_aug_layer = RandomBackgroundReplacement(
            prob=bg_aug_prob, face_center_ratio=bg_aug_face_ratio
        )
        train_ds = train_ds.map(
            lambda x, y: (bg_aug_layer(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    # 4.6) Clean possible decode/augment errors for train
    # (It is important to do this BEFORE adding weights)
    train_ds = train_ds.apply(tf.data.experimental.ignore_errors())

    # 4.75) GradCAM weights: load 
    gradcam_weights = None
    if gradcam_weights_path:
        if os.path.exists(gradcam_weights_path):
            try:
                with open(gradcam_weights_path, "r", encoding="utf-8") as f:
                    weight_data = json.load(f)

                weights_list = []
                train_root = os.path.join(base_dir, "train")
                for fp in train_file_paths:
                    rel_path = os.path.relpath(fp, train_root).replace("\\", "/")
                    # JSON'da yoksa 1.0 default
                    weights_list.append(weight_data.get(rel_path, 1.0))

                if weights_list:
                    gradcam_weights = tf.convert_to_tensor(
                        weights_list, dtype=tf.float32
                    )
                    min_w, max_w = gradcam_weight_clip
                    gradcam_weights = tf.clip_by_value(
                        gradcam_weights * gradcam_weight_scale,
                        clip_value_min=min_w,
                        clip_value_max=max_w,
                    )
                    print(
                        f"[GradCAM Weights] Loaded {len(weights_list)} entries from {gradcam_weights_path}"
                    )
            except Exception as exc:
                print(
                    f"[GradCAM Weights] ⚠️ Failed to load {gradcam_weights_path}: {exc}"
                )
                gradcam_weights = None
        else:
            print(f"[GradCAM Weights] ⚠️ File not found: {gradcam_weights_path}")

    # 5) Sample weights logic
    # ------------------------------------------------
    
    # CASE B: use_masks = False ve GradCAM JSON mevcut → GradCAM weights
    if gradcam_weights is not None:
        # 1) train_ds is currently batched: (batch, 224,224,3), (batch,1)
        #    First convert it to per-example
        train_ds = train_ds.unbatch()  # (x_single, y_single)

        # 2) Dataset for GradCAM weights: one weight per example
        gradcam_weight_ds = tf.data.Dataset.from_tensor_slices(gradcam_weights)  # (w,)

        # 3) Zip a single example with a single weight
        train_ds = tf.data.Dataset.zip((train_ds, gradcam_weight_ds))  # ((x,y), w)

        # 4) Unpack to (x, y, w) format
        def attach_gradcam_weights(data, w):
            x, y = data
            w = tf.cast(w, tf.float32)
            return x, y, w

        train_ds = train_ds.map(
            attach_gradcam_weights, num_parallel_calls=AUTOTUNE
        )  # (x_single, y_single, w_single)

        # 5) re-batch
        train_ds = train_ds.batch(batch_size, drop_remainder=False)

        print("[Sample Weights] Using GradCAM-derived weights from JSON")

    else:
        print("[Sample Weights] NOT using sample weights")

    # 6) Performance: cache + prefetch
    # (Extra shuffling on train helps)
    train_ds = train_ds.cache().shuffle(1000, seed=seed).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, list(class_names)
