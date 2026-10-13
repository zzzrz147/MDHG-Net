import tensorflow as tf


class ScaleSENet(tf.keras.layers.Layer):
    def __init__(self, channels, num_scales, reduction=4, dynamic=True):
        super().__init__()
        self.dynamic_weights = dynamic
        self.fc1 = tf.keras.layers.Dense(max(1, channels // reduction), activation='relu')
        self.fc2 = tf.keras.layers.Dense(1)
        self.static_logits = self.add_weight(name='static_logits', shape=(num_scales,),
                                             initializer='zeros', trainable=not dynamic)

    def call(self, states):
        descriptors = tf.stack([tf.reduce_mean(state, axis=2) for state in states], axis=2)
        context = tf.broadcast_to(tf.reduce_mean(descriptors, axis=2, keepdims=True), tf.shape(descriptors))
        if self.dynamic_weights:
            scores = tf.squeeze(self.fc2(self.fc1(tf.concat([descriptors, context], axis=-1))), axis=-1)
        else:
            scores = tf.broadcast_to(self.static_logits[None, None, :], tf.shape(descriptors)[:3])
        return tf.nn.softmax(scores, axis=-1), descriptors
