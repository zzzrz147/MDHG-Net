import csv
import json
from pathlib import Path
import numpy as np
import tensorflow as tf


def export_model_details(model, x, output_dir, sample_index=0):
    x = tf.convert_to_tensor(x)
    if not 0 <= sample_index < int(tf.shape(x)[0].numpy()):
        raise ValueError('sample_index is outside the batch.')
    previous = bool(model._mc_dropout_enabled.numpy())
    model._mc_dropout_enabled.assign(False)
    try:
        model(x, training=False, return_details=True, mc_dropout=False)
    finally:
        model._mc_dropout_enabled.assign(previous)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    weights = model.latest_scale_time_weights[sample_index].numpy()
    with (destination / 'scale_weights.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['anchor'] + ['window_{}h'.format(window) for window in model.windows])
        for anchor in range(weights.shape[1]):
            writer.writerow([anchor + 1] + weights[:, anchor].tolist())
    arrays = {'scale_weights': weights}
    for scale, detail in enumerate(model.latest_hypergraph_details):
        for key, value in detail.items():
            if tf.is_tensor(value):
                arrays['scale_{}_{}'.format(scale, key)] = value[sample_index].numpy()
    np.savez_compressed(destination / 'model_details.npz', **arrays)
    metadata = {'sample_index': sample_index, 'input_columns': model.feature_names,
                'role_names': model.role_names, 'scale_windows': model.windows,
                'incidence_shape': '[anchor,variable,candidate_hyperedge]',
                'node_semantics': 'variable', 'captured_with_dropout': False}
    (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata
