import math
import tensorflow as tf


def gaussian_nll(mu, variance, target, include_constant=False):
    mu = tf.convert_to_tensor(mu)
    variance = tf.cast(tf.convert_to_tensor(variance), mu.dtype)
    target = tf.cast(tf.convert_to_tensor(target), mu.dtype)
    tf.debugging.assert_equal(tf.shape(mu), tf.shape(variance))
    tf.debugging.assert_equal(tf.shape(mu), tf.shape(target))
    tf.debugging.assert_all_finite(variance, 'Variance must be finite.')
    tf.debugging.assert_positive(variance)
    loss = 0.5 * (tf.math.log(variance) + tf.square(target - mu) / variance)
    if include_constant:
        loss = loss + 0.5 * math.log(2.0 * math.pi)
    return tf.reduce_mean(loss)
