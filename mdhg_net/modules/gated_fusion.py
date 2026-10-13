import tensorflow as tf


class GatedFusion(tf.keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.gate_projection = tf.keras.layers.Dense(channels)

    def call(self, local, cross):
        gate = tf.nn.sigmoid(self.gate_projection(tf.concat([local, cross], axis=-1)))
        return gate * local + (1.0 - gate) * cross
