import copy
import tensorflow as tf

from .config import validate_config
from .modules import (
    MultiScaleAggregation, ScaleNormalization, TemporalEncoder, ScaleSENet,
    build_role_prior, DynamicHypergraph, PairwiseGraph, CrossScalePropagation,
    GatedFusion, ScaleFusion, TemporalDecoder, GaussianOutputHead,
)


class MDHGNet(tf.keras.Model):
    def __init__(self, config, feature_names=None, scale_means=None, scale_stds=None):
        super().__init__(name='mdhg_net')
        self.feature_names = validate_config(config, feature_names)
        self.construction_config = copy.deepcopy(config)
        self.seq_len = int(config['seq_len'])
        self.anchor_len = int(config.get('temporal_anchor_len', self.seq_len))
        self.mc_passes = int(config.get('mc_passes', 50))
        self.confidence_levels = tuple(config.get('confidence_levels', [0.9, 0.95]))
        self.windows = list(config['window_size'])
        self.horizons = list(config['forecast_horizons'])
        self.interaction = config.get('interaction', 'hypergraph')
        self.cross_scale = bool(config.get('cross_scale', True))
        self._mc_dropout_enabled = self.add_weight(name='mc_dropout_enabled', shape=(), dtype=tf.bool,
                                                  initializer=tf.keras.initializers.Constant(False), trainable=False)
        channels, dropout = int(config['d_model']), float(config['dropout'])
        self.aggregation = MultiScaleAggregation(self.windows, self.feature_names, config['energy_columns'],
                                                config.get('sampling_interval_hours', 1.0))
        self.normalization = ScaleNormalization(len(self.windows), len(self.feature_names), scale_means, scale_stds)
        self.use_fixed_statistics = self.normalization.use_fixed_statistics
        self.encoders = [TemporalEncoder(channels, dropout, config.get('encoder_kernel_size', 3),
                                         config.get('encoder_stride', 1), config.get('encoder_layers', 2),
                                         config.get('encoder_activation', 'relu'), config.get('interpolation_align_corners', False))
                         for _ in self.windows]
        self.scale_se = ScaleSENet(channels, len(self.windows), config.get('scale_se_reduction', 4),
                                 config.get('dynamic_scale_weights', True))
        self.role_names = list(config['role_groups'])
        self.hypergraphs = []
        if self.interaction == 'hypergraph':
            for edges, threshold in zip(config['hyper_num'], config['incidence_thresholds']):
                prior = build_role_prior(self.feature_names, config['role_groups'], edges)
                self.hypergraphs.append(DynamicHypergraph(channels, prior, threshold, dropout,
                                                         config.get('dynamic_hypergraph', True)))
        elif self.interaction == 'pairwise':
            self.hypergraphs = [PairwiseGraph(channels, dropout) for _ in self.windows]
        self.cross_propagation = CrossScalePropagation(channels, dropout)
        self.fusion = GatedFusion(channels)
        self.scale_fusion = ScaleFusion()
        self.temporal_decoder = TemporalDecoder(channels, int(config['gru_hidden']), dropout,
                                                config.get('gru_layers', 1), config.get('gru_dropout', 0.0))
        self.probabilistic_head = GaussianOutputHead(int(config['gru_hidden']), len(self.horizons),
                                                    config.get('variance_epsilon', 1e-6))
        self.latest_scale_time_weights = None
        self.latest_hypergraph_details = None

    @property
    def mean_head(self):
        return self.probabilistic_head.mean_head

    @property
    def variance_head(self):
        return self.probabilistic_head.variance_head

    def call(self, x, training=False, return_details=False, mc_dropout=None):
        x = tf.cast(tf.convert_to_tensor(x), tf.float32)
        tf.debugging.assert_rank(x, 3)
        tf.debugging.assert_equal(tf.shape(x)[1:], [self.seq_len, len(self.feature_names)])
        enabled = self._mc_dropout_enabled if mc_dropout is None else mc_dropout
        dropout_training = tf.logical_or(tf.cast(enabled, tf.bool), tf.cast(False if training is None else training, tf.bool))
        sequences = self.normalization(self.aggregation(x))
        states = [encoder(sequence, self.anchor_len, training=dropout_training)
                  for encoder, sequence in zip(self.encoders, sequences)]
        weights, descriptors = self.scale_se(states)
        weighted = [state * weights[:, :, s, None, None] for s, state in enumerate(states)]
        details = []
        if self.interaction == 'none':
            fused = weighted
        else:
            local, edges, incidence = [], [], []
            for s, module in enumerate(self.hypergraphs):
                result = (module(weighted[s], descriptors[:, :, s, :], training=dropout_training)
                          if self.interaction == 'hypergraph' else module(weighted[s], training=dropout_training))
                local.append(result[0])
                edges.append(result[1])
                incidence.append(result[2])
                if return_details:
                    item = dict(result[3])
                    item.update(incidence=result[2], node_semantics='variable')
                    details.append(item)
            if self.cross_scale:
                cross = self.cross_propagation(edges, incidence, training=dropout_training)
                fused = [self.fusion(within, across) for within, across in zip(local, cross)]
            else:
                fused = local
        sequence = self.scale_fusion(fused, weights)
        state = self.temporal_decoder(sequence, training=training, dropout_training=dropout_training)
        output = self.probabilistic_head(state)
        if return_details:
            self.latest_scale_time_weights = tf.stop_gradient(tf.transpose(weights, [0, 2, 1]))
            self.latest_hypergraph_details = [{key: tf.stop_gradient(value) if tf.is_tensor(value) else value
                                              for key, value in item.items()} for item in details]
        return output
