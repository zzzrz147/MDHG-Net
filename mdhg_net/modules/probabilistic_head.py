import tensorflow as tf


class GaussianOutputHead(tf.keras.layers.Layer):
    def __init__(self, hidden, num_horizons, variance_epsilon=1e-6):
        super().__init__()
        self.mean_head = tf.keras.layers.Dense(num_horizons)
        self.variance_head = tf.keras.layers.Dense(num_horizons)
        self.variance_epsilon = float(variance_epsilon)

    def call(self, predictive_state):
        mean = self.mean_head(predictive_state)[:, :, None]
        variance = tf.nn.softplus(self.variance_head(predictive_state))[:, :, None] + self.variance_epsilon
        return {'mu': mean, 'variance': variance, 'sigma': tf.sqrt(variance)}
