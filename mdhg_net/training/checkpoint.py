from pathlib import Path
import json
import tensorflow as tf
from ..model import MDHGNet


def resolve_device(device):
    value = str(device).lower()
    if value in ('cpu', '/cpu:0'):
        return '/CPU:0'
    if value.startswith(('gpu', 'cuda', '/gpu:')):
        index = int(value.rsplit(':', 1)[1]) if ':' in value else 0
        if index >= len(tf.config.list_physical_devices('GPU')):
            raise ValueError('The requested GPU is unavailable.')
        return '/GPU:' + str(index)
    raise ValueError('device must be cpu or gpu:index.')


def build_model(model):
    if not model.built:
        model(tf.zeros([1, model.seq_len, len(model.feature_names)], dtype=tf.float32), training=False)
    return model


def save_model(model, config, output_dir):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    build_model(model)
    path = destination / 'best_model.weights.h5'
    model.save_weights(str(path))
    metadata = {'config': config, 'feature_names': model.feature_names,
                'fixed_statistics': model.use_fixed_statistics,
                'framework': 'tensorflow', 'tensorflow_version': tf.__version__}
    if model.use_fixed_statistics:
        metadata['scale_means'] = model.normalization.scale_means.numpy().tolist()
        metadata['scale_stds'] = model.normalization.scale_stds.numpy().tolist()
    (destination / 'model_config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return path


def load_model(checkpoint_path, device='cpu'):
    path = Path(checkpoint_path)
    if path.is_dir():
        path = path / 'best_model.weights.h5'
    metadata = json.loads((path.parent / 'model_config.json').read_text(encoding='utf-8'))
    extra = {'scale_means': metadata['scale_means'], 'scale_stds': metadata['scale_stds']} if metadata['fixed_statistics'] else {}
    with tf.device(resolve_device(device)):
        model = MDHGNet(metadata['config'], metadata['feature_names'], **extra)
        build_model(model)
        model.load_weights(str(path))
    return model
