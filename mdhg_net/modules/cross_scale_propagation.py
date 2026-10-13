import math
import tensorflow as tf


class CrossScalePropagation(tf.keras.layers.Layer):
    def __init__(self, channels, dropout):
        super().__init__()
        self.query = tf.keras.layers.Dense(channels, use_bias=False)
        self.key = tf.keras.layers.Dense(channels, use_bias=False)
        self.value = tf.keras.layers.Dense(channels, use_bias=False)
        self.node_projection = tf.keras.layers.Dense(channels, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, edge_states, incidences, training=False):
        propagated = []
        for s, target in enumerate(edge_states):
            updated = target
            for r, source in enumerate(edge_states):
                if r == s:
                    continue
                attention = tf.nn.softmax(tf.matmul(self.query(target), self.key(source), transpose_b=True)
                                           / math.sqrt(target.shape[-1]), axis=-1)
                updated = updated + tf.matmul(attention, self.value(source))
            propagated.append(self.dropout(tf.nn.relu(self.node_projection(tf.matmul(incidences[s], updated))),
                                            training=training))
        return propagated
