import tensorflow as tf


class ScaleFusion(tf.keras.layers.Layer):
    def call(self, states, weights):
        summaries = tf.stack([tf.reduce_mean(state, axis=2) for state in states], axis=2)
        return tf.reduce_sum(summaries * weights[..., None], axis=2)
