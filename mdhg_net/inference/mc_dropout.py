from contextlib import contextmanager
from statistics import NormalDist
import math
import tensorflow as tf


@contextmanager
def mc_dropout_mode(model):
    previous = bool(model._mc_dropout_enabled.numpy())
    model._mc_dropout_enabled.assign(True)
    try:
        yield
    finally:
        model._mc_dropout_enabled.assign(previous)


def combine_mc_moments(means, variances):
    means, variances = tf.convert_to_tensor(means), tf.convert_to_tensor(variances)
    tf.debugging.assert_equal(tf.shape(means), tf.shape(variances))
    tf.debugging.assert_positive(tf.shape(means)[0])
    mean = tf.reduce_mean(means, axis=0)
    aleatoric = tf.reduce_mean(variances, axis=0)
    epistemic = tf.reduce_mean(tf.square(means - mean[None, ...]), axis=0)
    total = aleatoric + epistemic
    return {'mu': mean, 'variance': total, 'sigma': tf.sqrt(total),
            'aleatoric_variance': aleatoric, 'epistemic_variance': epistemic}


def mc_predict(model, x, passes=None, confidence_levels=None, retain_samples=False):
    passes = model.mc_passes if passes is None else passes
    confidence_levels = model.confidence_levels if confidence_levels is None else confidence_levels
    if not isinstance(passes, int) or passes < 1:
        raise ValueError('passes must be a positive integer.')
    means, variances = [], []
    with mc_dropout_mode(model):
        for _ in range(passes):
            output = model(x, training=False, mc_dropout=True)
            means.append(output['mu'])
            variances.append(output['variance'])
    means, variances = tf.stack(means), tf.stack(variances)
    result = combine_mc_moments(means, variances)
    result['intervals'] = {}
    for level in confidence_levels:
        if not 0 < level < 1:
            raise ValueError('Confidence levels must lie within (0,1).')
        z = NormalDist().inv_cdf((1.0 + level) / 2.0)
        result['intervals'][float(level)] = (result['mu'] - z * result['sigma'], result['mu'] + z * result['sigma'])
    if retain_samples:
        result['sample_means'], result['sample_variances'] = means, variances
    return result


def to_physical_units(prediction, target_mean=0.0, target_scale=1.0):
    if not math.isfinite(target_scale) or target_scale <= 0 or not math.isfinite(target_mean):
        raise ValueError('Target scale must be finite and positive.')
    converted = dict(prediction)
    for name in ('mu', 'sample_means'):
        if name in converted:
            converted[name] = converted[name] * target_scale + target_mean
    for name in ('variance', 'aleatoric_variance', 'epistemic_variance', 'sample_variances'):
        if name in converted:
            converted[name] = converted[name] * target_scale ** 2
    if 'sigma' in converted:
        converted['sigma'] = converted['sigma'] * target_scale
    if 'intervals' in converted:
        converted['intervals'] = {level: (lower * target_scale + target_mean, upper * target_scale + target_mean)
                                  for level, (lower, upper) in converted['intervals'].items()}
    return converted
