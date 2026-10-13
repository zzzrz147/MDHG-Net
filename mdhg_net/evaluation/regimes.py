import numpy as np
import tensorflow as tf
from .metrics import forecast_metrics


def historical_volatility(generation, load, history_hours=72):
    generation = tf.convert_to_tensor(generation)
    load = tf.cast(tf.convert_to_tensor(load), generation.dtype)
    tf.debugging.assert_rank(generation, 2)
    tf.debugging.assert_equal(tf.shape(generation), tf.shape(load))
    tf.debugging.assert_greater_equal(tf.shape(generation)[1], history_hours + 1)
    residual = generation[:, -(history_hours + 1):] - load[:, -(history_hours + 1):]
    changes = residual[:, 1:] - residual[:, :-1]
    return tf.sqrt(tf.reduce_mean(tf.square(changes), axis=1))


def fit_regime_boundaries(training_volatility, quantiles=(1/3, 2/3)):
    values = np.asarray(training_volatility, np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError('Training volatility must be finite and nonempty.')
    if len(quantiles) != 2 or not 0 < quantiles[0] < quantiles[1] < 1:
        raise ValueError('Provide two increasing quantiles within (0,1).')
    return tf.constant(np.quantile(values, quantiles), dtype=tf.float64)


def regime_metrics(mu, variance, target, test_volatility, training_boundaries):
    mu, variance, target = [np.asarray(value) for value in (mu, variance, target)]
    volatility = np.asarray(test_volatility).reshape(-1)
    boundaries = np.asarray(training_boundaries).reshape(-1)
    if volatility.size != mu.shape[0] or boundaries.size != 2:
        raise ValueError('Invalid sample count or regime boundaries.')
    groups = np.searchsorted(boundaries, volatility, side='left')
    result = {}
    for index, name in enumerate(('low', 'moderate', 'high')):
        mask = groups == index
        result[name] = {'samples': int(mask.sum()),
                        'metrics': forecast_metrics(mu[mask], variance[mask], target[mask]) if mask.any() else None}
    return result
