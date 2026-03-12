import tensorflow as tf
from typing import Optional


class SimpleMaskedLoss(tf.keras.losses.Loss):
    def __init__(self, name="simple_masked_loss", from_logits: bool = False, **kwargs):
        # Keep reduction as NONE and compute the mean ourselves
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE, **kwargs)
        self.from_logits = from_logits

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Per-element BCE, shape (batch, 1) or (batch,)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        ce = tf.keras.losses.binary_crossentropy(
            y_true, y_pred, from_logits=self.from_logits
        )
        return ce  # We do not reduce yet

    def __call__(
        self,
        y_true,
        y_pred,
        sample_weight: Optional[tf.Tensor] = None,
        **kwargs,
    ):
        ce = self.call(y_true, y_pred, **kwargs)  # (batch, 1) or (batch,)

        # Squeeze → (batch,)
        if tf.rank(ce) > 1:
            ce = tf.squeeze(ce, axis=-1)
        ce = tf.cast(ce, tf.float32)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            # Flatten and align lengths to the minimum (for safety)
            ce = tf.reshape(ce, [-1])
            sample_weight = tf.reshape(sample_weight, [-1])

            n = tf.minimum(tf.shape(ce)[0], tf.shape(sample_weight)[0])
            ce = ce[:n]
            sample_weight = sample_weight[:n]

            tf.print(
                "[MaskedLoss] ✅ sample_weight RECEIVED | loss_len:",
                tf.shape(ce)[0],
                "| sw_len:", tf.shape(sample_weight)[0],
                "| mean:", tf.reduce_mean(sample_weight),
                "| min:", tf.reduce_min(sample_weight),
                "| max:", tf.reduce_max(sample_weight),
            )

            total_weight = tf.reduce_sum(sample_weight) + 1e-8
            weighted = ce * sample_weight
            loss = tf.reduce_sum(weighted) / total_weight
        else:
            tf.print(
                "[MaskedLoss] ⚠️ sample_weight is None | loss_len:",
                tf.shape(ce)[0],
            )
            loss = tf.reduce_mean(ce)

        return loss


def create_simple_masked_loss():
    return SimpleMaskedLoss()
