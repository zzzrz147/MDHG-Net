import numpy as np
import tensorflow as tf


class BatchLoader:
    def __init__(self, source, config, phase):
        self.source_dataset = source
        self.batch_size = int(config['batch_size'])
        self.drop_last = phase == 'train' and bool(config.get('train_drop_last', False))
        self.shuffle = phase == 'train'
        self.phase = phase
        if isinstance(source, tf.data.Dataset):
            dataset = source
            spec = source.element_spec
            x_spec = spec['x'] if isinstance(spec, dict) else spec[0]
            if x_spec.shape.rank == 3:
                dataset = dataset.unbatch()
            elif x_spec.shape.rank != 2:
                raise ValueError('Prepared samples must have shape [history,variables].')
        elif isinstance(source, (tuple, list)) and len(source) == 2:
            x, y = source
            dataset = tf.data.Dataset.from_tensor_slices((tf.cast(tf.convert_to_tensor(x), tf.float32),
                                                         tf.cast(tf.convert_to_tensor(y), tf.float32)))
        else:
            raise TypeError('Provide a tf.data.Dataset or a prepared (x,y) pair.')
        cardinality = int(tf.data.experimental.cardinality(dataset).numpy())
        if cardinality == tf.data.experimental.INFINITE_CARDINALITY:
            raise ValueError('Dataset must be finite.')
        if self.shuffle:
            dataset = dataset.shuffle(max(1, cardinality) if cardinality >= 0 else 10000,
                                      seed=int(config.get('seed', 2025)), reshuffle_each_iteration=True)
        options = tf.data.Options()
        options.experimental_deterministic = True
        options.threading.private_threadpool_size = max(1, int(config.get('num_workers', 0)))
        dataset = dataset.with_options(options)
        self.dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_last)

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        count = int(tf.data.experimental.cardinality(self.dataset).numpy())
        if count < 0:
            raise TypeError('The dataset cardinality is unknown.')
        return count


def make_loaders(train_dataset, val_dataset, test_dataset, config):
    return (BatchLoader(train_dataset, config, 'train'), BatchLoader(val_dataset, config, 'val'),
            BatchLoader(test_dataset, config, 'test'))


def check_loader_settings(loader, config, phase):
    actual = getattr(loader, 'batch_size', getattr(loader, '_mdhg_batch_size', None))
    expected = int(config['batch_size'])
    if actual is not None and actual != expected:
        raise ValueError('Dataset batch size differs from the configuration.')
    if actual is None and config.get('enforce_paper_settings', False):
        raise ValueError('Use make_loaders() or attach _mdhg_batch_size to the batched dataset.')
    if phase in ('val', 'test') and (getattr(loader, 'drop_last', False) or getattr(loader, 'shuffle', False)):
        raise ValueError('Validation and test datasets must retain all samples in chronological order.')
    source = getattr(loader, 'source_dataset', None)
    if hasattr(source, 'forecast_horizons') and list(source.forecast_horizons) != list(config['forecast_horizons']):
        raise ValueError('Dataset forecast horizons differ from the configuration.')
    if hasattr(source, 'feature_names') and list(source.feature_names) != list(config['input_columns']):
        raise ValueError('Dataset feature order differs from the configuration.')
    return {'batch_size': actual, 'drop_last': getattr(loader, 'drop_last', None),
            'shuffle': getattr(loader, 'shuffle', None), 'backend': 'tf.data'}
