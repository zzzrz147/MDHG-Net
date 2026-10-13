import tensorflow as tf


def inverse_degree(degree, power):
    return tf.where(degree > 0, tf.pow(tf.maximum(degree, 1e-12), power), tf.zeros_like(degree))


class HypergraphConvolution(tf.keras.layers.Layer):
    def __init__(self, channels, num_edges, dropout):
        super().__init__()
        self.node_proj = tf.keras.layers.Dense(channels, use_bias=False)
        self.edge_weight = self.add_weight(name='edge_weight', shape=(num_edges, num_edges),
                                           initializer=tf.keras.initializers.Identity())
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, states, incidence, training=False):
        node_inv = inverse_degree(tf.reduce_sum(incidence, axis=-1), -0.5)
        edge_inv = inverse_degree(tf.reduce_sum(incidence, axis=-2), -1.0)
        projected = self.node_proj(states) * node_inv[..., None]
        edge_messages = tf.matmul(incidence, projected, transpose_a=True) * edge_inv[..., None]
        edge_messages = tf.matmul(self.edge_weight, edge_messages)
        local = self.dropout(tf.nn.relu(tf.matmul(incidence, edge_messages) * node_inv[..., None]), training=training)
        edge_states = tf.matmul(incidence, local, transpose_a=True) * edge_inv[..., None]
        return local, edge_states
