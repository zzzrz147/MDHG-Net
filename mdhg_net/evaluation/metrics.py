import math
from statistics import NormalDist
import numpy as np
import tensorflow as tf


def normal_cdf(z):
    return 0.5 * tf.math.erfc(-tf.convert_to_tensor(z) / math.sqrt(2.0))


def gaussian_crps(mu, variance, target):
    mu = tf.convert_to_tensor(mu)
    variance = tf.cast(tf.convert_to_tensor(variance), mu.dtype)
    target = tf.cast(tf.convert_to_tensor(target), mu.dtype)
    sigma = tf.sqrt(variance)
    z = (target - mu) / sigma
    density = tf.exp(-0.5 * tf.square(z)) / math.sqrt(2.0 * math.pi)
    return sigma * (z * (2.0 * normal_cdf(z) - 1.0) + 2.0 * density - 1.0 / math.sqrt(math.pi))


def forecast_metrics(mu, variance, target, thresholds=(500.0, 600.0, 700.0),
                     confidence_levels=(0.9, 0.95), alarm_probability=0.5):
    mu, variance, target = [np.asarray(value, dtype=np.float64) for value in (mu, variance, target)]
    if mu.shape != target.shape or variance.shape != target.shape or not target.size:
        raise ValueError('Provide matching, nonempty prediction, variance, and target shapes.')
    if any(not np.isfinite(value).all() for value in (mu, variance, target)) or (variance <= 0).any():
        raise ValueError('Metrics require finite values and positive variance.')
    if not 0 <= alarm_probability <= 1 or any(not 0 < level < 1 for level in confidence_levels):
        raise ValueError('Invalid probability or confidence level.')
    mu, variance, target = [value.reshape(-1) for value in (mu, variance, target)]
    error, sigma = target - mu, np.sqrt(variance)
    crps = gaussian_crps(tf.constant(mu), tf.constant(variance), tf.constant(target)).numpy()
    nll = 0.5 * (math.log(2.0 * math.pi) + np.log(variance) + error ** 2 / variance)
    centered_true, centered_pred = target - target.mean(), mu - mu.mean()
    denominator = np.sqrt(np.sum(centered_true ** 2) * np.sum(centered_pred ** 2))
    nonzero = np.abs(target) > 1e-6
    result = {'samples': int(target.size), 'MAE': float(np.abs(error).mean()),
              'RMSE': float(np.sqrt(np.mean(error ** 2))),
              'MAPE_percent': float(100 * np.abs(error[nonzero] / target[nonzero]).mean()) if nonzero.any() else None,
              'MAPE_excluded_near_zero': int((~nonzero).sum()),
              'Corr': float(np.dot(centered_true, centered_pred) / denominator) if denominator > 0 else None,
              'CRPS': float(crps.mean()), 'NLL': float(nll.mean()), 'AvgSigma': float(sigma.mean()),
              'distribution_used_for_scores': 'moment_matched_Gaussian'}
    width_range = float(np.max(target) - np.min(target))
    for level in confidence_levels:
        z = NormalDist().inv_cdf((1.0 + level) / 2.0)
        result['PICP@{}'.format(int(100 * level))] = float((np.abs(error) <= z * sigma).mean())
        result['PINAW@{}'.format(int(100 * level))] = float(2 * z * sigma.mean() / width_range) if width_range > 0 else None
    result['low_balance'] = {}
    for threshold in thresholds:
        observed = target <= threshold
        probability = normal_cdf(tf.constant((threshold - mu) / sigma)).numpy()
        predicted = probability >= alarm_probability
        tp = int((observed & predicted).sum())
        fp = int((~observed & predicted).sum())
        fn = int((observed & ~predicted).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result['low_balance'][str(threshold)] = {
            'events': int(observed.sum()), 'event_ratio': float(observed.mean()),
            'MAE': float(np.abs(error[observed]).mean()) if observed.any() else None,
            'CRPS': float(crps[observed].mean()) if observed.any() else None,
            'precision': precision, 'recall': recall,
            'F1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'Brier': float(np.mean((probability - observed.astype(float)) ** 2)),
            'true_positive': tp, 'false_positive': fp, 'false_negative': fn}
    return result


def reliability_curve(mu, variance, target, levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)):
    mu, variance, target = [np.asarray(value, dtype=np.float64) for value in (mu, variance, target)]
    sigma = np.sqrt(variance)
    return [{'nominal': level, 'empirical': float((np.abs(target - mu) <= NormalDist().inv_cdf((1 + level) / 2) * sigma).mean())}
            for level in levels]
