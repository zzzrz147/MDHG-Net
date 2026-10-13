from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path

from mdhg_net.experiments import experiment_configs, run_repeated
from mdhg_net.paper_protocol import audit_config


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(root/'configs/default.json'))
    parser.add_argument('--suite', choices=('main','horizons','scales','ablations'), default='main')
    parser.add_argument('--factory', help='Existing function module:function taking (config,seed) and returning three loaders')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', default='outputs/paper_runs')
    parser.add_argument('--target-mean', type=float, default=0.0)
    parser.add_argument('--target-scale', type=float, default=1.0)
    parser.add_argument('--describe', action='store_true', help='Inspect plans and parameter provenance without training')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    cases = experiment_configs(config, args.suite)
    if args.describe:
        for name, case in cases.items():
            print(json.dumps({'experiment': name, 'forecast_horizons': case['forecast_horizons'],
                              'window_size': case['window_size'], 'runs': case['repetitions'],
                              'parameter_audit': audit_config(case)}, indent=2, ensure_ascii=False))
        return
    if not args.factory or ':' not in args.factory:
        parser.error('Supply --factory module:function from your existing data code, or use --describe.')
    module, function = args.factory.split(':',1)
    loader_factory = getattr(importlib.import_module(module), function)
    for name, case in cases.items():
        run_repeated(case, loader_factory, device=args.device, output_dir=Path(args.output_dir)/name,
                     target_mean=args.target_mean, target_scale=args.target_scale)


if __name__ == '__main__':
    main()
