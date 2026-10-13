from __future__ import annotations
import math
from .paper_protocol import validate_paper_settings


def validate_config(config, feature_names=None):
    names = list(feature_names if feature_names is not None else config['input_columns'])
    windows, horizons = list(config['window_size']), list(config['forecast_horizons'])
    if int(config['seq_len']) < 1 or not windows or any(int(w) != w or w < 1 for w in windows):
        raise ValueError('seq_len and absolute window lengths must be positive integers.')
    if sorted(set(windows)) != windows or windows[0] != 1:
        raise ValueError('window_size must start at 1 and contain increasing absolute windows.')
    if not horizons or any(int(h) != h or h < 1 for h in horizons):
        raise ValueError('forecast_horizons must contain positive integer lead times.')
    if len(set(horizons)) != len(horizons):
        raise ValueError('forecast_horizons must not contain duplicate lead times.')
    if not names or len(set(names)) != len(names):
        raise ValueError('Input feature names must be nonempty and unique.')
    if config.get('target', 'balance') in names:
        raise ValueError('Historical balance must not be included among input columns (Eq. 1).')
    if config.get('interaction', 'hypergraph') not in {'hypergraph', 'pairwise', 'none'}:
        raise ValueError('interaction must be hypergraph, pairwise, or none.')
    if int(config['d_model']) < 1 or int(config['gru_hidden']) < 1 or not 0 <= float(config['dropout']) < 1:
        raise ValueError('Hidden dimensions must be positive and dropout must lie in [0,1).')
    if int(config.get('scale_se_reduction', 4)) < 1:
        raise ValueError('scale_se_reduction must be positive.')
    if float(config.get('variance_epsilon', 1e-6)) <= 0:
        raise ValueError('variance_epsilon must be positive.')
    if not set(config['energy_columns']).issubset(set(names)):
        raise ValueError('energy_columns must be a subset of input_columns.')
    groups = list(config['role_groups'].values())
    if not groups or any(not isinstance(group, list) for group in groups):
        raise ValueError('role_groups must map role names to lists of variable names.')
    if any(name not in names for group in groups for name in group):
        raise ValueError('role_groups contains a variable absent from input_columns.')
    edges, thresholds = list(config['hyper_num']), list(config['incidence_thresholds'])
    if len(edges) != len(windows) or len(thresholds) != len(windows):
        raise ValueError('hyper_num and incidence_thresholds must have one entry per scale.')
    if any(int(m) != m or m < 1 for m in edges) or any(not 0 <= eta < 1 for eta in thresholds):
        raise ValueError('Hyperedge counts must be positive integers; thresholds must lie in [0,1).')
    kernel = config.get('encoder_kernel_size', 3)
    stride = config.get('encoder_stride', 1)
    layers = config.get('encoder_layers', 2)
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in (kernel, stride, layers)) or kernel % 2 != 1:
        raise ValueError('Encoder kernel must be odd; kernel, stride, and layers must be positive integers.')
    finest_length = int(config['seq_len'])
    for _ in range(layers):
        finest_length = (finest_length + 2*(kernel//2) - kernel)//stride + 1
    if int(config.get('temporal_anchor_len', config['seq_len'])) != finest_length:
        raise ValueError('temporal_anchor_len must equal the encoded finest-scale length (Eq. 10).')
    for name in ('mc_passes', 'batch_size', 'train_epochs', 'patience', 'gru_layers'):
        if name in config and (not isinstance(config[name], int) or isinstance(config[name], bool) or config[name] < 1):
            raise ValueError(f'{name} must be a positive integer.')
    for name in ('learning_rate', 'optimizer_eps', 'sampling_interval_hours'):
        value = float(config.get(name, {'learning_rate': 0.001, 'optimizer_eps': 1e-8, 'sampling_interval_hours': 1.0}[name]))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive.')
    if config.get('optimizer', 'AdamW').lower() != 'adamw' or config.get('loss', 'gaussian_nll') != 'gaussian_nll':
        raise ValueError('The implemented optimizer/loss are AdamW and gaussian_nll.')
    if config.get('lr_scheduler', 'none') != 'none':
        raise ValueError('No learning-rate scheduler is reported or enabled in this implementation.')
    betas = config.get('optimizer_betas', [0.9, 0.999])
    if len(betas) != 2 or any(not 0 <= beta < 1 for beta in betas):
        raise ValueError('optimizer_betas must contain two values in [0,1).')
    if not math.isfinite(float(config['weight_decay'])) or config['weight_decay'] < 0:
        raise ValueError('weight_decay must be finite and nonnegative.')
    if config.get('early_stop_min_delta', 0.0) < 0:
        raise ValueError('early_stop_min_delta must be nonnegative.')
    ratios = [config.get(name, fallback) for name, fallback in [('train_ratio',0.8),('val_ratio',0.1),('test_ratio',0.1)]]
    if any(not 0 < ratio < 1 for ratio in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-12):
        raise ValueError('Train/validation/test ratios must be positive and sum to one.')
    if config.get('enforce_paper_settings', False):
        validate_paper_settings(config)
    return names
