from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import numpy as np
import tensorflow as tf

from .model import MDHGNet
from .config import validate_config
from .paper_protocol import PAPER_HORIZONS, PAPER_SCALE_SETS, audit_config
from .training import fit, predict_loader, set_seed, check_loader_settings
from .inference import to_physical_units
from .evaluation import forecast_metrics


def experiment_configs(base_config, suite='main', thresholds_by_scale_count=None):






    cases = {}
    if suite == 'main':
        if len(base_config['forecast_horizons']) != 1:
            raise ValueError('The main experiment must specify one forecast lead.')
        cases[f"full_h{base_config['forecast_horizons'][0]}"] = copy.deepcopy(base_config)
    elif suite == 'horizons':
        for lead in PAPER_HORIZONS:
            config = copy.deepcopy(base_config)
            config.update(forecast_horizons=[lead], window_size=[1,12,24],
                          incidence_thresholds=[0.05,0.1,0.15], hyper_num=[6,6,6])
            cases[f'lead_{lead}h'] = config
    elif suite == 'scales':
        for windows in PAPER_SCALE_SETS:
            config = copy.deepcopy(base_config)
            count = len(windows)
            supplied = (thresholds_by_scale_count or {}).get(count)
            thresholds = supplied if supplied is not None else [0.05,0.1,0.15][:count] + [0.15]*max(0, count-3)
            config.update(forecast_horizons=[12], window_size=list(windows), hyper_num=[6]*count,
                          incidence_thresholds=list(thresholds))
            if count != 3:
                config['implementation_notes'] = ['Thresholds for this scale count are not numerically reported in the manuscript. '
                                                  'Use author-provided values to reproduce the original experiment.']
            cases['scales_' + '_'.join(map(str,windows))] = config
    elif suite == 'ablations':
        variants = {
            'full': {}, 'static_scale_weights': {'dynamic_scale_weights': False},
            'static_hypergraph': {'dynamic_hypergraph': False},
            'no_cross_scale': {'cross_scale': False}, 'pairwise_graph': {'interaction': 'pairwise'},
            'no_structured_interaction': {'interaction': 'none'},
        }
        for name, changes in variants.items():
            config = copy.deepcopy(base_config)
            config.update(forecast_horizons=[12]); config.update(changes)
            cases[name] = config
    else:
        raise ValueError('suite must be main, horizons, scales, or ablations.')
    for config in cases.values():
        validate_config(config)
    return cases


def _flatten_numbers(value, prefix=''):
    output = {}
    for name, item in value.items():
        key = f'{prefix}.{name}' if prefix else str(name)
        if isinstance(item, dict):
            output.update(_flatten_numbers(item, key))
        elif isinstance(item, (int,float)) and not isinstance(item,bool) and math.isfinite(item):
            output[key] = float(item)
    return output


def summarize_runs(metrics):

    records = [_flatten_numbers(item) for item in metrics]
    keys = sorted(set.intersection(*(set(record) for record in records))) if records else []
    summary = {}
    for key in keys:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        summary[key] = {'mean': float(values.mean()),
                        'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        'runs': len(values)}
    return summary


def run_repeated(config, loader_factory, device='cpu', output_dir=None, model_factory=None,
                 target_mean=0.0, target_scale=1.0):







    validate_config(config)
    seeds = list(config['experiment_seeds'])
    repetitions = int(config['repetitions'])
    if repetitions != len(seeds) or repetitions < 1 or len(set(seeds)) != repetitions:
        raise ValueError('repetitions must equal the number of distinct experiment_seeds.')
    if len(config['forecast_horizons']) != 1:
        raise ValueError('Paper experiments train each forecast lead in a separate run.')
    destination = Path(output_dir) if output_dir is not None else None
    runs, all_metrics = [], []
    for seed in seeds:
        run_config = copy.deepcopy(config); run_config['seed'] = int(seed)
        tf.keras.backend.clear_session()
        set_seed(int(seed))
        loaders = loader_factory(run_config, int(seed))
        if len(loaders) != 3:
            raise ValueError('loader_factory must return train, validation, and test loaders.')
        train_loader, val_loader, test_loader = loaders
        check_loader_settings(test_loader, run_config, 'test')
        model = model_factory(run_config) if model_factory is not None else MDHGNet(run_config)
        run_directory = destination / f'seed_{seed}' if destination is not None else None
        training = fit(model, train_loader, val_loader, run_config, device, run_directory)
        mu, variance, target = predict_loader(model, test_loader, device=device)
        physical = to_physical_units({'mu':mu, 'variance':variance}, target_mean, target_scale)
        target = target * target_scale + target_mean
        metrics = forecast_metrics(physical['mu'], physical['variance'], target,
                                   thresholds=config['low_balance_thresholds'],
                                   confidence_levels=config['confidence_levels'],
                                   alarm_probability=config['alarm_probability'])
        runs.append({'seed': int(seed), 'training': training, 'metrics': metrics})
        all_metrics.append(metrics)
        if run_directory is not None:
            (run_directory / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
        del model
    result = {'repetitions_completed': len(runs), 'seeds': seeds, 'forecast_horizons': config['forecast_horizons'],
              'reported_units': 'MW when caller supplies MW targets or the correct affine conversion',
              'standard_deviation_definition': 'sample, ddof=1; paper does not specify ddof',
              'parameter_audit': audit_config(config), 'runs': runs, 'aggregate': summarize_runs(all_metrics)}
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result
