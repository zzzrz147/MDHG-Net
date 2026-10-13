import tensorflow as tf


def aggregate_windows(x, window, energy_mask, sampling_interval_hours=1.0):
    x = tf.convert_to_tensor(x)
    shape = tf.shape(x)
    padding = tf.math.floormod(-shape[1], window)
    pads = tf.stack([tf.constant([0, 0]), tf.stack([padding, 0]), tf.constant([0, 0])])
    x = tf.pad(x, pads)
    blocks = tf.reshape(x, [shape[0], -1, window, shape[2]])
    summed = tf.reduce_sum(blocks, axis=2)
    mask = tf.cast(energy_mask, tf.bool)[None, None, :]
    return tf.where(mask, summed * sampling_interval_hours, summed / tf.cast(window, x.dtype))


class MultiScaleAggregation(tf.keras.layers.Layer):
    def __init__(self, windows, feature_names, energy_columns, sampling_interval_hours=1.0):
        super().__init__()
        self.windows = tuple(int(window) for window in windows)
        self.sampling_interval_hours = float(sampling_interval_hours)
        energy = set(energy_columns)
        self.energy_mask = self.add_weight(name='energy_mask', shape=(len(feature_names),), dtype=tf.bool,
                                          initializer=tf.keras.initializers.Constant([name in energy for name in feature_names]),
                                          trainable=False)

    def call(self, x):
        return [aggregate_windows(x, window, self.energy_mask, self.sampling_interval_hours) for window in self.windows]
