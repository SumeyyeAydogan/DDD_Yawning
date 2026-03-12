"""
Adversarial augmentation strategies to reduce background dependency.
"""

import tensorflow as tf
import numpy as np


class BackgroundAugmentation(tf.keras.layers.Layer):
    """
    Data augmentation layer that randomizes background to force model
    to focus on face region instead of background.
    
    Strategies:
    1. Random background color replacement
    2. Random background blur
    3. Random background noise
    4. Random background texture overlay
    """
    
    def __init__(self, 
                 background_replace_prob=0.3,
                 background_blur_prob=0.2,
                 background_noise_prob=0.2,
                 **kwargs):
        """
        Args:
            background_replace_prob: Probability of replacing background with solid color
            background_blur_prob: Probability of blurring background
            background_noise_prob: Probability of adding noise to background
        """
        super().__init__(**kwargs)
        self.background_replace_prob = background_replace_prob
        self.background_blur_prob = background_blur_prob
        self.background_noise_prob = background_noise_prob
        
    def call(self, images, training=None):
        """
        Apply background augmentation during training.
        
        Args:
            images: Batch of images (B, H, W, C)
            training: Whether in training mode
            
        Returns:
            Augmented images
        """
        if not training:
            return images
        
        # Use tf.map_fn to process batch in graph mode
        def process_single_image(img):
            return self._augment_single_image(img)
        
        # Process entire batch using vectorized_map for better performance
        augmented_images = tf.vectorized_map(process_single_image, images)
        return augmented_images
    
    def _augment_single_image(self, image):
        """Apply random background augmentation to single image"""
        # Assume face is in center 40% of image
        H = tf.shape(image)[0]
        W = tf.shape(image)[1]
        C = tf.shape(image)[2]
        
        center_h = tf.cast(tf.cast(H, tf.float32) * 0.4, tf.int32)
        center_w = tf.cast(tf.cast(W, tf.float32) * 0.4, tf.int32)
        
        y_start = (H - center_h) // 2
        y_end = y_start + center_h
        x_start = (W - center_w) // 2
        x_end = x_start + center_w
        
        # Create mask: 1.0 for face region, 0.0 for background
        # Use coordinate-based approach for dynamic shapes
        y_coords = tf.range(H, dtype=tf.float32)
        x_coords = tf.range(W, dtype=tf.float32)
        Y, X = tf.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Create mask: 1.0 if inside center region, 0.0 otherwise
        y_mask = tf.logical_and(Y >= tf.cast(y_start, tf.float32), 
                                Y < tf.cast(y_end, tf.float32))
        x_mask = tf.logical_and(X >= tf.cast(x_start, tf.float32), 
                                X < tf.cast(x_end, tf.float32))
        face_mask = tf.logical_and(y_mask, x_mask)
        face_mask = tf.cast(face_mask, tf.float32)
        face_mask = tf.expand_dims(face_mask, -1)  # (H, W, 1)
        face_mask = tf.tile(face_mask, [1, 1, C])  # (H, W, C)
        
        # Random augmentation strategy - use tf.cond instead of Python if
        rand = tf.random.uniform([], 0.0, 1.0)
        
        # Define augmentation functions
        def replace_background():
            bg_color = tf.random.uniform([C], 0.0, 1.0)
            background = tf.ones_like(image) * bg_color
            return image * face_mask + background * (1 - face_mask)
        
        def blur_background():
            noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.1)
            blurred_bg = tf.clip_by_value(image + noise, 0.0, 1.0)
            return image * face_mask + blurred_bg * (1 - face_mask)
        
        def noise_background():
            noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.15)
            noisy_bg = tf.clip_by_value(image + noise, 0.0, 1.0)
            return image * face_mask + noisy_bg * (1 - face_mask)
        
        def no_augmentation():
            return image
        
        # Use nested tf.cond for multiple conditions
        threshold1 = self.background_replace_prob
        threshold2 = self.background_replace_prob + self.background_blur_prob
        threshold3 = self.background_replace_prob + self.background_blur_prob + self.background_noise_prob
        
        # First condition: replace background
        augmented = tf.cond(
            rand < threshold1,
            replace_background,
            lambda: tf.cond(
                rand < threshold2,
                blur_background,
                lambda: tf.cond(
                    rand < threshold3,
                    noise_background,
                    no_augmentation
                )
            )
        )
        
        return augmented
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'background_replace_prob': self.background_replace_prob,
            'background_blur_prob': self.background_blur_prob,
            'background_noise_prob': self.background_noise_prob
        })
        return config


def create_background_augmented_dataset(dataset, 
                                       background_replace_prob=0.3,
                                       background_blur_prob=0.2,
                                       background_noise_prob=0.2):
    """
    Create dataset with background augmentation applied.
    
    Args:
        dataset: tf.data.Dataset
        background_replace_prob: Probability of replacing background
        background_blur_prob: Probability of blurring background
        background_noise_prob: Probability of adding noise to background
        
    Returns:
        Augmented dataset
    """
    aug_layer = BackgroundAugmentation(
        background_replace_prob=background_replace_prob,
        background_blur_prob=background_blur_prob,
        background_noise_prob=background_noise_prob
    )
    
    def augment_fn(x, y):
        # Apply augmentation only to images
        x_aug = aug_layer(x, training=True)
        return x_aug, y
    
    return dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)


class RandomBackgroundReplacement(tf.keras.layers.Layer):
    """
    Simpler version: Just replace background with random colors/textures.
    More aggressive but easier to implement.
    """
    
    def __init__(self, prob=0.4, face_center_ratio=0.75, **kwargs):
        super().__init__(**kwargs)
        self.prob = prob
        self.face_center_ratio = face_center_ratio
    
    def call(self, images, training=None):
        if not training:
            return images
        
        # Use tf.map_fn to process batch in graph mode
        def process_single_image(img):
            # Random decision for this image
            should_augment = tf.random.uniform([], 0.0, 1.0) < self.prob
            
            # Conditional: apply augmentation or return original
            augmented = tf.cond(
                should_augment,
                lambda: self._replace_background(img),
                lambda: img
            )
            return augmented
        
        # Process entire batch using vectorized_map for better performance
        augmented_images = tf.vectorized_map(process_single_image, images)
        return augmented_images
    
    def _replace_background(self, image):
        """Replace background with random color"""
        H = tf.shape(image)[0]
        W = tf.shape(image)[1]
        C = tf.shape(image)[2]
        
        # Calculate center region
        center_h = tf.cast(tf.cast(H, tf.float32) * self.face_center_ratio, tf.int32)
        center_w = tf.cast(tf.cast(W, tf.float32) * self.face_center_ratio, tf.int32)
        
        y_start = (H - center_h) // 2
        y_end = y_start + center_h
        x_start = (W - center_w) // 2
        x_end = x_start + center_w
        
        # Create face mask: 1.0 in center, 0.0 elsewhere
        # Use coordinate-based approach for dynamic shapes
        y_coords = tf.range(H, dtype=tf.float32)
        x_coords = tf.range(W, dtype=tf.float32)
        Y, X = tf.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Create mask: 1.0 if inside center region, 0.0 otherwise
        y_mask = tf.logical_and(Y >= tf.cast(y_start, tf.float32), 
                                Y < tf.cast(y_end, tf.float32))
        x_mask = tf.logical_and(X >= tf.cast(x_start, tf.float32), 
                                X < tf.cast(x_end, tf.float32))
        face_mask = tf.logical_and(y_mask, x_mask)
        face_mask = tf.cast(face_mask, tf.float32)
        face_mask = tf.expand_dims(face_mask, -1)  # (H, W, 1)
        face_mask = tf.tile(face_mask, [1, 1, C])  # (H, W, C)
        
        # Random background color
        bg_color = tf.random.uniform([C], 0.0, 1.0)
        background = tf.ones_like(image) * bg_color
        
        # Combine: keep face, replace background
        return image * face_mask + background * (1 - face_mask)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'prob': self.prob,
            'face_center_ratio': self.face_center_ratio
        })
        return config

