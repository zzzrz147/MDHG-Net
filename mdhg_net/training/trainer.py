import csv
import json
import random
from pathlib import Path
import numpy as np
import tensorflow as tf

from ..config import validate_config
from ..losses import gaussian_nll
from ..inference import mc_predict
from .adamw import AdamW
from .checkpoint import build_model, save_model, load_model, resolve_device
from .loaders import check_loader_settings


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def unpack_xy(batch, device='cpu'):
    x, y = (batch['x'], batch['y']) if isinstance(batch, dict) else batch[:2]
    with tf.device(resolve_device(device)):
        return tf.cast(tf.convert_to_tensor(x), tf.float32), tf.cast(tf.convert_to_tensor(y), tf.float32)


def run_epoch(model, loader, device='cpu', optimizer=None, grad_clip=None):
    weighted_loss, elements = 0.0, 0
    for batch in loader:
        x, y = unpack_xy(batch, device)
        tf.debugging.assert_all_finite(x, 'Inputs must be finite.')
        tf.debugging.assert_all_finite(y, 'Targets must be finite.')
        with tf.device(resolve_device(device)):
            if optimizer is not None:
                with tf.GradientTape() as tape:
                    output = model(x, training=True)
                    loss = gaussian_nll(output['mu'], output['variance'], y)
                gradients = tape.gradient(loss, model.trainable_variables)
                pairs = [(gradient, variable) for gradient, variable in zip(gradients, model.trainable_variables)
                         if gradient is not None]
                if not pairs:
                    raise RuntimeError('No trainable gradients were produced.')
                for gradient, _ in pairs:
                    value = gradient.values if isinstance(gradient, tf.IndexedSlices) else gradient
                    tf.debugging.assert_all_finite(value, 'Gradients must be finite.')
                if grad_clip is not None:
                    clipped, _ = tf.clip_by_global_norm([pair[0] for pair in pairs], float(grad_clip))
                    pairs = list(zip(clipped, [pair[1] for pair in pairs]))
                tf.debugging.assert_all_finite(loss, 'NLL must be finite.')
                optimizer.apply_gradients(pairs)
            else:
                output = model(x, training=False)
                loss = gaussian_nll(output['mu'], output['variance'], y)
                tf.debugging.assert_all_finite(loss, 'NLL must be finite.')
        count = int(tf.size(y).numpy())
        weighted_loss += float(loss.numpy()) * count
        elements += count
    if not elements:
        raise ValueError('Dataset yielded no samples.')
    return weighted_loss / elements


def fit(model, train_loader, val_loader, config, device='cpu', output_dir=None):
    validate_config(config)
    fields = ('seq_len', 'temporal_anchor_len', 'forecast_horizons', 'window_size', 'd_model', 'gru_hidden',
              'dropout', 'hyper_num', 'incidence_thresholds', 'dynamic_scale_weights', 'dynamic_hypergraph',
              'cross_scale', 'interaction', 'input_columns', 'energy_columns', 'role_groups', 'mc_passes',
              'confidence_levels', 'encoder_kernel_size', 'encoder_stride', 'encoder_layers', 'encoder_activation',
              'gru_layers', 'gru_dropout', 'sampling_interval_hours', 'variance_epsilon')
    mismatches = [name for name in fields if model.construction_config.get(name) != config.get(name)]
    if mismatches:
        raise ValueError('Model and training configuration differ: ' + ', '.join(mismatches))
    train_settings = check_loader_settings(train_loader, config, 'train')
    val_settings = check_loader_settings(val_loader, config, 'val')
    epochs, patience = int(config['train_epochs']), int(config['patience'])
    seed = int(config.get('seed', 2025))
    if epochs < 1 or patience < 1 or seed < 0:
        raise ValueError('Invalid epoch, patience, or seed value.')
    set_seed(seed)
    with tf.device(resolve_device(device)):
        build_model(model)
        betas = config.get('optimizer_betas', [0.9, 0.999])
        optimizer = AdamW(learning_rate=float(config['learning_rate']), weight_decay=float(config['weight_decay']),
                          beta_1=betas[0], beta_2=betas[1], epsilon=float(config.get('optimizer_eps', 1e-8)))
    history, best_weights, best_loss, best_epoch, stale = [], None, float('inf'), 0, 0
    min_delta = float(config.get('early_stop_min_delta', 0.0))
    for epoch in range(1, epochs + 1):
        train_nll = run_epoch(model, train_loader, device, optimizer, config.get('grad_clip'))
        val_nll = run_epoch(model, val_loader, device)
        lr = float(tf.keras.backend.get_value(optimizer.learning_rate))
        history.append({'epoch': epoch, 'learning_rate': lr, 'train_nll': train_nll, 'val_nll': val_nll})
        print('Epoch {:03d}: train_nll={:.6f}, val_nll={:.6f}'.format(epoch, train_nll, val_nll))
        if val_nll < best_loss - min_delta:
            best_loss, best_epoch, stale = val_nll, epoch, 0
            best_weights = [value.copy() for value in model.get_weights()]
        else:
            stale += 1
        if stale >= patience:
            break
    model.set_weights(best_weights)
    summary = {'best_epoch': best_epoch, 'best_val_nll': best_loss, 'epochs_completed': len(history),
               'training_loss_constant_included': False, 'history': history, 'seed': seed, 'run_scope': 'single_seed',
               'framework': 'tensorflow', 'tensorflow_version': tf.__version__,
               'effective_settings': {'optimizer': 'AdamW', 'learning_rate': float(config['learning_rate']),
                                      'weight_decay': float(config['weight_decay']), 'optimizer_betas': list(betas),
                                      'optimizer_eps': optimizer.epsilon, 'maximum_epochs': epochs,
                                      'early_stop_patience': patience, 'early_stop_monitor': 'val_nll',
                                      'early_stop_min_delta': min_delta, 'learning_rate_scheduler': 'none',
                                      'train_loader': train_settings, 'val_loader': val_settings,
                                      'forecast_horizons': model.horizons, 'temporal_anchor_len': model.anchor_len,
                                      'mc_passes': model.mc_passes,
                                      'normalization': 'fixed_training_statistics' if model.use_fixed_statistics else 'scale_layer_norm'}}
    if output_dir is not None:
        destination = Path(output_dir)
        save_model(model, config, destination)
        (destination / 'training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        with (destination / 'training_history.csv').open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=['epoch', 'learning_rate', 'train_nll', 'val_nll'])
            writer.writeheader()
            writer.writerows(history)
    return summary


def predict_loader(model, loader, device='cpu', passes=None):
    means, variances, targets = [], [], []
    with tf.device(resolve_device(device)):
        for batch in loader:
            x, y = unpack_xy(batch, device)
            output = mc_predict(model, x, passes=passes)
            means.append(output['mu'])
            variances.append(output['variance'])
            targets.append(y)
    if not means:
        raise ValueError('Dataset yielded no samples.')
    return tf.concat(means, axis=0), tf.concat(variances, axis=0), tf.concat(targets, axis=0)
