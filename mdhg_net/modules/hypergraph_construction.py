import math
import tensorflow as tf


class HypergraphConstruction(tf.keras.layers.Layer):
    def __init__(self, channels, prior, threshold, dynamic=True):
        super().__init__()
        prior = tf.convert_to_tensor(prior, tf.float32)
        nodes, edges = prior.shape
        self.prior = self.add_weight(name='prior', shape=(nodes, edges),
                                     initializer=tf.keras.initializers.Constant(prior.numpy()), trainable=False)
        self.prototypes = self.add_weight(name='prototypes', shape=(edges, channels), initializer='glorot_uniform')
        self.query = tf.keras.layers.Dense(channels, use_bias=False)
        self.edge_proj = tf.keras.layers.Dense(channels, use_bias=False)
        self.prior_gate = tf.keras.layers.Dense(edges)
        self.threshold = float(threshold)
        self.dynamic_incidence = dynamic
        self.static_scores = self.add_weight(name='static_scores', shape=(nodes, edges),
                                             initializer=tf.keras.initializers.Constant(prior.numpy()), trainable=not dynamic)

    def call(self, states, descriptor):
        if self.dynamic_incidence:
            scores = tf.matmul(self.query(states), self.edge_proj(self.prototypes), transpose_b=True)
            scores = scores / math.sqrt(states.shape[-1])
            gate = tf.nn.sigmoid(self.prior_gate(descriptor))[:, :, None, :]
            scores = gate * scores + (1.0 - gate) * self.prior[None, None, :, :]
        else:
            scores = tf.broadcast_to(self.static_scores[None, None, :, :],
                                     tf.concat([tf.shape(states)[:2], tf.shape(self.static_scores)], axis=0))
            gate = None
        soft = tf.nn.softmax(scores, axis=-1)
        incidence = tf.where(soft >= self.threshold, soft, tf.zeros_like(soft))
        return incidence, {'soft_incidence': soft, 'prior_gate': gate}
