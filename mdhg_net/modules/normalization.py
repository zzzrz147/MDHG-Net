import numpy as np
import tensorflow as tf


class ScaleNormalization(tf.keras.layers.Layer):
    def __init__(self, num_scales, num_variables, means=None, stds=None):
        super().__init__()
        self.use_fixed_statistics = means is not None or stds is not None
        if self.use_fixed_statistics:
            if means is None or stds is None:
                raise ValueError('Provide both scale_means and scale_stds.')
            means, stds = np.asarray(means, np.float32), np.asarray(stds, np.float32)
            expected = (num_scales, num_variables)
            if means.shape != expected or stds.shape != expected:
                raise ValueError('Scale statistics have an invalid shape.')
            if not np.isfinite(means).all() or not np.isfinite(stds).all() or (stds <= 0).any():
                raise ValueError('Scale statistics must be finite with positive standard deviations.')
            self.scale_means = self.add_weight(name='scale_means', shape=expected,
                                               initializer=tf.keras.initializers.Constant(means), trainable=False)
            self.scale_stds = self.add_weight(name='scale_stds', shape=expected,
                                              initializer=tf.keras.initializers.Constant(stds), trainable=False)
        else:
            self.normalizers = [tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5) for _ in range(num_scales)]

    def call(self, sequences):
        if self.use_fixed_statistics:
            return [(x - self.scale_means[s]) / self.scale_stds[s] for s, x in enumerate(sequences)]
        return [self.normalizers[s](x) for s, x in enumerate(sequences)]
