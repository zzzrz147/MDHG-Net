from __future__ import annotations

import math


PAPER_SETTINGS = {
    'sampling_interval_hours': (1.0, 'Eq. (6), Table 2'),
    'seq_len': (72, 'Section 4.1, Table 2'),
    'temporal_anchor_len': (72, 'Section 4.1, Table 2'),
    'd_model': (64, 'Section 4.1, Table 2'),
    'dropout': (0.3, 'Section 4.1, Table 2'),
    'optimizer': ('AdamW', 'Section 4.1, Table 2'),
    'learning_rate': (0.001, 'Section 4.1, Table 2'),
    'weight_decay': (0.001, 'Section 4.1, Table 2'),
    'batch_size': (64, 'Section 4.1, Table 2'),
    'train_epochs': (200, 'Section 4.1, Table 2; maximum, not mandatory completion'),
    'patience': (20, 'Section 4.1, Table 2'),
    'loss': ('gaussian_nll', 'Eqs. (43)-(44), Table 2'),
    'mc_passes': (50, 'Section 4.1, Table 2'),
    'repetitions': (5, 'Section 4.1'),
    'train_ratio': (0.8, 'Table 2; applied by external dataset split'),
    'val_ratio': (0.1, 'Table 2; applied by external dataset split'),
    'test_ratio': (0.1, 'Table 2; applied by external dataset split'),
    'confidence_levels': ([0.9, 0.95], 'Section 4.1, Table 4'),
    'low_balance_thresholds': ([500.0, 600.0, 700.0], 'Section 4.4, Table 7'),
    'low_balance_reference': (600.0, 'Section 4.4'),
    'alarm_probability': (0.5, 'Section 4.4'),
    'volatility_history_hours': (72, 'Section 4.5, Eq. (45)'),
    'volatility_quantiles': ([1/3, 2/3], 'Section 4.5'),
}
PAPER_HORIZONS = [12, 24, 48, 72]
PAPER_SCALE_SETS = [[1,12], [1,12,24], [1,12,24,48], [1,12,24,48,72]]
IMPLEMENTATION_CHOICES = {
    'encoder_kernel_size': 'Convolution kernel size is not stated.',
    'encoder_stride': 'Convolution stride is not stated; 1 retains the finest resolution.',
    'encoder_layers': 'The number of convolution layers is not stated.',
    'encoder_activation': 'The nonlinear activation delta is not named.',
    'interpolation_align_corners': 'Linear interpolation is stated; align_corners is not.',
    'gru_hidden': 'A separate GRU hidden size is not stated; it is set to the latent size.',
    'gru_layers': 'The number of GRU layers is not stated.',
    'gru_dropout': 'Internal GRU dropout is not separately stated.',
    'scale_se_reduction': 'Excitation bottleneck size is not stated.',
    'hyper_num': 'The number of candidate hyperedges at each scale is not stated.',
    'variance_epsilon': 'Epsilon is positive; its numerical value is not stated.',
    'optimizer_betas': 'AdamW beta values are not stated; standard defaults are used.',
    'optimizer_eps': 'AdamW epsilon is not stated; the standard default is used.',
    'lr_scheduler': 'Only initial learning rate is stated; no scheduler is specified.',
    'early_stop_min_delta': 'Early-stopping minimum improvement is not stated.',
    'grad_clip': 'Gradient clipping is not stated and is disabled by default.',
    'num_workers': 'dataset batch worker count is not stated.',
    'pin_memory': 'dataset batch pin_memory is not stated.',
    'train_drop_last': 'Handling of the last incomplete batch is not stated.',
    'experiment_seeds': 'Five seeds are required; their numerical values are not stated.',
}


def same_value(actual, expected):
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(
            same_value(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)
    return actual == expected


def audit_config(config):
    rows = []
    for name, (expected, source) in PAPER_SETTINGS.items():
        actual = config.get(name)
        status = 'matches' if same_value(actual, expected) else 'differs'
        if name in {'train_ratio', 'val_ratio', 'test_ratio'} and status == 'matches':
            status = 'external_contract'
        rows.append({'setting': name, 'configured': actual, 'paper': expected,
                     'source': source, 'status': status})
    horizons = config.get('forecast_horizons', [])
    rows.append({'setting': 'forecast_horizons', 'configured': horizons, 'paper': PAPER_HORIZONS,
                 'source': 'Sections 4.1 and 4.3; train each lead separately',
                 'status': 'matches' if len(horizons) == 1 and horizons[0] in PAPER_HORIZONS else 'differs'})
    windows = config.get('window_size', [])
    rows.append({'setting': 'window_size', 'configured': windows, 'paper': PAPER_SCALE_SETS,
                 'source': 'Sections 4.1 and 4.3', 'status': 'matches' if windows in PAPER_SCALE_SETS else 'differs'})
    rows.append({'setting': 'incidence_thresholds', 'configured': config.get('incidence_thresholds'),
                 'paper': [0.05,0.10,0.15], 'source': 'Section 4.1, default three-scale setup',
                 'status': ('matches' if same_value(config.get('incidence_thresholds'), [0.05,0.1,0.15]) else 'differs')
                           if windows == [1,12,24] else 'implementation_choice'})
    for name, reason in IMPLEMENTATION_CHOICES.items():
        rows.append({'setting': name, 'configured': config.get(name), 'paper': None,
                     'source': reason, 'status': 'implementation_choice'})
    rows.extend([
        {'setting': 'input_variable_count', 'configured': len(config['input_columns']), 'paper': 23,
         'source': 'Section 3.1, Table 2; provided-column contract retained', 'status': 'user_override'},
        {'setting': 'normalization', 'configured': 'scale LayerNorm or caller-supplied training statistics',
         'paper': 'Norm, unspecified operator', 'source': 'Eq. (7)', 'status': 'implementation_choice'},
        {'setting': 'normalization_epsilon', 'configured': 1e-5, 'paper': None,
         'source': 'LayerNorm standard default; the manuscript does not specify its epsilon', 'status': 'implementation_choice'},
        {'setting': 'zero_padding_side', 'configured': 'oldest end, left', 'paper': None,
         'source': 'Eq. (4) defines padded length but not padding direction; no padding occurs in the default scale set', 'status': 'implementation_choice'},
        {'setting': 'parameter_initialization', 'configured': 'TensorFlow/Keras layer defaults, Xavier-uniform edge prototypes',
         'paper': None, 'source': 'Initialization distributions are not reported', 'status': 'implementation_choice'},
        {'setting': 'sample_window_stride', 'configured': 'external Dataset', 'paper': None,
         'source': 'No sliding-sample stride is specified; dataset creation is outside this package', 'status': 'external_contract'},
        {'setting': 'hyperedge_weight_matrix', 'configured': 'full learnable Ms x Ms, identity initialization',
         'paper': 'Wh in R^(Ms x Ms)', 'source': 'Eq. (23); initialization is an implementation choice', 'status': 'matches'},
        {'setting': 'hyperedge_weight_initialization', 'configured': 'identity', 'paper': None,
         'source': 'Eq. (23) does not give initialization values', 'status': 'implementation_choice'},
        {'setting': 'probabilistic_score_distribution', 'configured': 'moment-matched Gaussian', 'paper': None,
         'source': 'Eq. (42) states Gaussian interval approximation; MC-mixture score details are not specified', 'status': 'implementation_choice'},
        {'setting': 'MAPE_near_zero_handling', 'configured': 'exclude abs(target)<=1e-6', 'paper': None,
         'source': 'MAPE is supplementary; the exact denominator treatment is not specified', 'status': 'implementation_choice'},
        {'setting': 'PINAW_normalizer', 'configured': 'evaluated target maximum minus minimum', 'paper': None,
         'source': 'The exact PINAW normalization definition is not reported', 'status': 'implementation_choice'},
        {'setting': 'reported_seed_standard_deviation', 'configured': 'sample standard deviation, ddof=1', 'paper': None,
         'source': 'Mean and standard deviation are reported, but ddof is not specified', 'status': 'implementation_choice'},
    ])
    return rows


def validate_paper_settings(config):
    differences = [row for row in audit_config(config) if row['status'] == 'differs']
    if differences:
        detail = ', '.join(f"{row['setting']}={row['configured']}" for row in differences)
        raise ValueError('Configuration differs from reported paper settings: ' + detail +
                         '. Use enforce_paper_settings=false only for an explicitly customized run.')
    seeds = config.get('experiment_seeds', [])
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError('The paper reporting protocol requires five distinct experiment seeds.')
