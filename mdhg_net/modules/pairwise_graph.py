import math
import tensorflow as tf


class PairwiseGraph(tf.keras.layers.Layer):
    def __init__(self, channels, dropout):
        super().__init__()
        self.query = tf.keras.layers.Dense(channels, use_bias=False)
        self.key = tf.keras.layers.Dense(channels, use_bias=False)
        self.value = tf.keras.layers.Dense(channels, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, states, training=False):
        attention = tf.nn.softmax(tf.matmul(self.query(states), self.key(states), transpose_b=True)
                                  / math.sqrt(states.shape[-1]), axis=-1)
        local = self.dropout(tf.nn.relu(tf.matmul(attention, self.value(states))), training=training)
        nodes = tf.shape(states)[-2]
        identity = tf.broadcast_to(tf.eye(nodes, dtype=states.dtype),
                                   tf.concat([tf.shape(states)[:2], [nodes, nodes]], axis=0))
        return local, local, identity, {'pairwise_attention': attention}
