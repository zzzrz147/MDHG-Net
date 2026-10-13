import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import tensorflow as tf

from mdhg_net import MDHGNet
from mdhg_net.modules import aggregate_windows, DynamicHypergraph, CrossScalePropagation
from mdhg_net.modules.hypergraph_convolution import HypergraphConvolution
from mdhg_net.losses import gaussian_nll
from mdhg_net.inference import mc_predict, combine_mc_moments, mc_dropout_mode, to_physical_units
from mdhg_net.training import fit, make_loaders, load_model, set_seed, AdamW
from mdhg_net.training.checkpoint import save_model
from mdhg_net.experiments import run_repeated, experiment_configs, summarize_runs
from mdhg_net.paper_protocol import audit_config, validate_paper_settings
from mdhg_net.evaluation import forecast_metrics, historical_volatility, fit_regime_boundaries, regime_metrics
from mdhg_net.interpretability import export_model_details


class TensorFlowModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default = json.loads((Path(__file__).parents[1] / 'configs/default.json').read_text(encoding='utf-8'))

    def setUp(self):
        tf.keras.backend.clear_session()
        set_seed(42)

    def compact(self):
        config = copy.deepcopy(self.default)
        config.update(seq_len=12, temporal_anchor_len=12, window_size=[1, 4, 8], d_model=8, gru_hidden=8,
                      train_epochs=2, patience=2, batch_size=4, mc_passes=3, enforce_paper_settings=False)
        return config

    def test_exact_requested_tensorflow_version(self):
        self.assertEqual(tf.__version__, '2.9.1')

    def test_window_aggregation_and_left_padding(self):
        x = tf.constant([[[2., 3.], [4., 6.], [8., 9.], [16., 12.]]])
        result = aggregate_windows(x, 3, tf.constant([True, False]))
        np.testing.assert_allclose(result.numpy(), [[[2., 1.], [28., 9.]]])

    def test_gaussian_nll_uses_variance(self):
        value = gaussian_nll(tf.constant([0.]), tf.constant([4.]), tf.constant([2.]))
        self.assertAlmostEqual(float(value.numpy()), 0.5 * math.log(4) + 0.5, places=6)

    def test_full_model_shapes_and_gradients(self):
        model = MDHGNet(self.default)
        x = tf.random.normal([2, 72, 24])
        with tf.GradientTape() as tape:
            output = model(x, return_details=True)
            loss = gaussian_nll(output['mu'], output['variance'], tf.ones([2, 1, 1]))
        gradients = tape.gradient(loss, model.trainable_variables)
        self.assertEqual(tuple(output['mu'].shape), (2, 1, 1))
        self.assertTrue(np.all(output['variance'].numpy() > 0))
        self.assertEqual(tuple(model.latest_scale_time_weights.shape), (2, 3, 72))
        np.testing.assert_allclose(tf.reduce_sum(model.latest_scale_time_weights, axis=1), np.ones((2, 72)), atol=1e-6)
        self.assertTrue(any(value is not None and np.any(value.numpy() != 0) for value in gradients))
        by_identity = {id(variable): gradient for variable, gradient in zip(model.trainable_variables, gradients)}
        for variable in (model.mean_head.kernel, model.variance_head.kernel,
                         model.scale_se.fc2.kernel, model.hypergraphs[0].convolution.edge_weight):
            self.assertGreater(float(tf.reduce_sum(tf.abs(by_identity[id(variable)])).numpy()), 0)

    def test_dynamic_weights_and_incidence_depend_on_input(self):
        model = MDHGNet(self.compact())
        x = tf.random.normal([2, 12, 24])
        model(x, return_details=True)
        weights = model.latest_scale_time_weights.numpy().copy()
        incidence = model.latest_hypergraph_details[0]['soft_incidence'].numpy().copy()
        model(tf.roll(x, shift=3, axis=1), return_details=True)
        self.assertGreater(np.abs(weights - model.latest_scale_time_weights.numpy()).sum(), 0)
        self.assertGreater(np.abs(incidence - model.latest_hypergraph_details[0]['soft_incidence'].numpy()).sum(), 0)

    def test_threshold_retains_soft_membership_without_renormalization(self):
        model = MDHGNet(self.compact())
        model(tf.random.normal([2, 12, 24]), return_details=True)
        for index, details in enumerate(model.latest_hypergraph_details):
            soft = details['soft_incidence'].numpy()
            threshold = self.default['incidence_thresholds'][index]
            np.testing.assert_allclose(soft.sum(axis=-1), 1.0, atol=1e-6)
            np.testing.assert_allclose(details['incidence'].numpy(), soft * (soft >= threshold))

    def test_equation_23_full_edge_matrix(self):
        layer = HypergraphConvolution(2, 2, 0.0)
        h = tf.constant([[0.4, 0.2], [0.1, 0.6]])
        x = tf.constant([[1., 3.], [2., 4.]])
        wh = tf.constant([[1., 0.25], [0.5, 1.]])
        layer(x[None, None], h[None, None])
        layer.node_proj.kernel.assign(tf.eye(2))
        layer.edge_weight.assign(wh)
        with tf.GradientTape() as tape:
            result, _ = layer(x[None, None], h[None, None])
            loss = tf.reduce_sum(result)
        dv = tf.linalg.diag(tf.pow(tf.reduce_sum(h, axis=1), -0.5))
        de = tf.linalg.diag(tf.pow(tf.reduce_sum(h, axis=0), -1.0))
        expected = tf.nn.relu(dv @ h @ wh @ de @ tf.transpose(h) @ dv @ x)
        np.testing.assert_allclose(result.numpy()[0, 0], expected.numpy(), rtol=1e-6)
        self.assertGreater(abs(float(tape.gradient(loss, layer.edge_weight)[0, 1].numpy())), 0)

    def test_empty_incidence_is_finite(self):
        layer = DynamicHypergraph(2, tf.zeros([2, 2]), 0.99, 0.0, dynamic=False)
        local, edges, incidence, _ = layer(tf.ones([1, 1, 2, 2]), tf.zeros([1, 1, 2]))
        self.assertEqual(float(tf.reduce_sum(tf.abs(local)).numpy()), 0)
        self.assertEqual(float(tf.reduce_sum(tf.abs(edges)).numpy()), 0)
        self.assertTrue(np.isfinite(local.numpy()).all())

    def test_cross_scale_exchange_uses_other_scales(self):
        layer = CrossScalePropagation(2, 0.0)
        edges = [tf.ones([1, 3, 2, 2]), tf.ones([1, 3, 4, 2])]
        incidence = [tf.ones([1, 3, 5, 2]), tf.ones([1, 3, 5, 4])]
        layer(edges, incidence)
        for projection in (layer.query, layer.key, layer.value, layer.node_projection):
            projection.kernel.assign(tf.eye(2))
        np.testing.assert_allclose(layer(edges, incidence)[0], 4.0)
        np.testing.assert_allclose(layer([edges[0], edges[1] * 3], incidence)[0], 8.0)

    def test_scale_counts_and_multiple_leads(self):
        for windows in ([1, 12], [1, 12, 24, 48], [1, 12, 24, 48, 72]):
            config = copy.deepcopy(self.default)
            config.update(window_size=windows, hyper_num=[6] * len(windows), incidence_thresholds=[0.1] * len(windows),
                          forecast_horizons=[12, 24, 48, 72], enforce_paper_settings=False)
            output = MDHGNet(config)(tf.random.normal([1, 72, 24]))
            self.assertEqual(tuple(output['mu'].shape), (1, 4, 1))

    def test_all_ablation_modes_have_finite_gradients(self):
        for change in ({'dynamic_scale_weights': False}, {'dynamic_hypergraph': False}, {'cross_scale': False},
                       {'interaction': 'pairwise'}, {'interaction': 'none'}):
            config = self.compact()
            config.update(change)
            model = MDHGNet(config)
            with tf.GradientTape() as tape:
                output = model(tf.random.normal([2, 12, 24]), training=True)
                loss = gaussian_nll(output['mu'], output['variance'], tf.zeros([2, 1, 1]))
            gradients = tape.gradient(loss, model.trainable_variables)
            self.assertTrue(np.isfinite(loss.numpy()))
            self.assertTrue(all(np.isfinite(value.numpy()).all() for value in gradients if value is not None))

    def test_graph_execution_matches_eager(self):
        model = MDHGNet(self.compact())
        x = tf.random.normal([2, 12, 24])
        expected = model(x)
        compiled = tf.function(lambda value: model(value, training=False))
        actual = compiled(x)
        np.testing.assert_allclose(actual['mu'], expected['mu'], atol=1e-6, rtol=1e-5)

    def test_mc_moments_and_unit_conversion(self):
        result = combine_mc_moments(tf.constant([[1.], [3.]]), tf.constant([[4.], [4.]]))
        self.assertEqual(float(result['mu'].numpy()[0]), 2)
        self.assertEqual(float(result['variance'].numpy()[0]), 5)
        converted = to_physical_units(result, 10, 3)
        self.assertEqual(float(converted['mu'].numpy()[0]), 16)
        self.assertEqual(float(converted['variance'].numpy()[0]), 45)

    def test_mc_dropout_is_stochastic_and_restores_state(self):
        model = MDHGNet(self.compact())
        x = tf.random.normal([3, 12, 24])
        np.testing.assert_array_equal(model(x)['mu'], model(x)['mu'])
        prediction = mc_predict(model, x, passes=8)
        self.assertGreater(float(tf.reduce_mean(prediction['epistemic_variance']).numpy()), 0)
        self.assertFalse(bool(model._mc_dropout_enabled.numpy()))
        np.testing.assert_allclose(prediction['variance'], prediction['aleatoric_variance'] + prediction['epistemic_variance'])
        with self.assertRaises(RuntimeError):
            with mc_dropout_mode(model):
                raise RuntimeError('context interruption')
        self.assertFalse(bool(model._mc_dropout_enabled.numpy()))

    def test_graph_mc_context_changes_only_dropout(self):
        model = MDHGNet(self.compact())
        x = tf.random.normal([2, 12, 24])
        model(x)
        compiled = tf.function(lambda value: model(value, training=False)['mu'])
        deterministic = compiled(x).numpy()
        with mc_dropout_mode(model):
            changed = compiled(x).numpy()
        self.assertGreater(np.abs(deterministic - changed).sum(), 0)
        np.testing.assert_allclose(compiled(x), deterministic, atol=1e-6)

    def test_mc_count_and_confidence_read_from_configuration(self):
        config = self.compact()
        config.update(mc_passes=3, confidence_levels=[0.8])
        model = MDHGNet(config)
        model(tf.ones([1, 12, 24]))
        with patch.object(model, 'call', wraps=model.call) as call:
            output = mc_predict(model, tf.ones([1, 12, 24]))
        self.assertEqual(call.call_count, 3)
        self.assertEqual(set(output['intervals']), {0.8})

    def test_adamw_decay_and_bias_correction(self):
        variable = tf.Variable([1.0, -2.0])
        optimizer = AdamW(learning_rate=0.001, weight_decay=0.1)
        optimizer.apply_gradients([(tf.constant([2.0, -0.5]), variable)])
        np.testing.assert_allclose(variable.numpy(), [0.9989, -1.9988], atol=2e-7)

    def test_adamw_zero_gradient_still_decays_weights(self):
        variable = tf.Variable([2.0])
        optimizer = AdamW(learning_rate=0.01, weight_decay=0.1)
        optimizer.apply_gradients([(tf.zeros([1]), variable)])
        self.assertAlmostEqual(float(variable.numpy()[0]), 1.998, places=6)

    def test_adamw_runs_inside_tf_function(self):
        variable = tf.Variable([1.0])
        optimizer = AdamW()
        def update():
            with tf.GradientTape() as tape:
                loss = tf.reduce_sum(variable ** 2)
            optimizer.apply_gradients([(tape.gradient(loss, variable), variable)])
            return loss
        operation = tf.function(update)
        operation()
        operation()
        self.assertLess(float(variable.numpy()[0]), 1.0)
        self.assertEqual(int(optimizer.iterations.numpy()), 2)

    def test_training_updates_weights_and_reloads_checkpoint(self):
        config = self.compact()
        arrays = (np.random.normal(size=(8, 12, 24)).astype('float32'), np.random.normal(size=(8, 1, 1)).astype('float32'))
        train, val, _ = make_loaders(arrays, arrays, arrays, config)
        model = MDHGNet(config)
        model(arrays[0][:1])
        before = model.mean_head.kernel.numpy().copy()
        with tempfile.TemporaryDirectory() as directory:
            summary = fit(model, train, val, config, output_dir=directory)
            self.assertTrue(math.isfinite(summary['best_val_nll']))
            self.assertGreater(np.abs(before - model.mean_head.kernel.numpy()).sum(), 0)
            restored = load_model(directory)
            np.testing.assert_allclose(restored(arrays[0])['mu'], model(arrays[0])['mu'], atol=1e-7)
            export_model_details(restored, arrays[0], Path(directory) / 'details')
            self.assertTrue((Path(directory) / 'details/model_details.npz').exists())

    def test_fixed_normalization_and_ablation_checkpoint_reloads(self):
        for extra, change in (({'scale_means': np.zeros((3, 24)), 'scale_stds': np.ones((3, 24))}, {}),
                              ({}, {'dynamic_hypergraph': False}), ({}, {'interaction': 'pairwise'}), ({}, {'interaction': 'none'})):
            config = self.compact()
            config.update(change)
            model = MDHGNet(config, **extra)
            x = tf.random.normal([2, 12, 24])
            expected = model(x)
            with tempfile.TemporaryDirectory() as directory:
                save_model(model, config, directory)
                restored = load_model(directory)
                np.testing.assert_allclose(restored(x)['variance'], expected['variance'], atol=1e-7)

    def test_loader_preserves_tail_and_accepts_tf_dataset(self):
        config = self.compact()
        config['batch_size'] = 3
        source = tf.data.Dataset.from_tensor_slices((tf.ones([7, 12, 24]), tf.ones([7, 1, 1])))
        _, validation, test = make_loaders(source, source, source, config)
        self.assertEqual(sum(int(tf.shape(y)[0]) for _, y in validation), 7)
        self.assertEqual(sum(int(tf.shape(y)[0]) for _, y in test), 7)

    def test_loader_wrong_batch_size_is_rejected(self):
        config = self.compact()
        arrays = (np.zeros((4, 12, 24), 'float32'), np.zeros((4, 1, 1), 'float32'))
        wrong_config = copy.deepcopy(config)
        wrong_config['batch_size'] = 2
        train, val, _ = make_loaders(arrays, arrays, arrays, wrong_config)
        with self.assertRaises(ValueError):
            fit(MDHGNet(config), train, val, config)

    def test_early_stopping_and_best_epoch(self):
        config = self.compact()
        config.update(train_epochs=8, patience=2)
        arrays = (np.zeros((4, 12, 24), 'float32'), np.zeros((4, 1, 1), 'float32'))
        train, val, _ = make_loaders(arrays, arrays, arrays, config)
        with patch('mdhg_net.training.trainer.run_epoch', side_effect=[2., 5., 2., 4., 2., 4., 2., 4.]):
            summary = fit(MDHGNet(config), train, val, config)
        self.assertEqual(summary['best_epoch'], 2)
        self.assertEqual(summary['epochs_completed'], 4)

    def test_default_paper_parameters_and_unknown_choices(self):
        validate_paper_settings(self.default)
        self.assertFalse([row for row in audit_config(self.default) if row['status'] == 'differs'])
        changed = copy.deepcopy(self.default)
        changed['learning_rate'] = 0.01
        with self.assertRaises(ValueError):
            MDHGNet(changed)

    def test_independent_horizon_scale_and_ablation_plans(self):
        plans = experiment_configs(self.default, 'horizons')
        self.assertEqual([case['forecast_horizons'] for case in plans.values()], [[12], [24], [48], [72]])
        self.assertEqual(len(experiment_configs(self.default, 'scales')), 4)
        self.assertEqual(len(experiment_configs(self.default, 'ablations')), 6)

    def test_five_seed_pipeline_and_sample_std(self):
        config = self.compact()
        config.update(train_epochs=1, patience=1, mc_passes=2)
        calls = []
        def factory(settings, seed):
            calls.append(seed)
            generator = np.random.RandomState(123)
            arrays = (generator.normal(size=(4, 12, 24)).astype('float32'), generator.normal(size=(4, 1, 1)).astype('float32'))
            return make_loaders(arrays, arrays, arrays, settings)
        with tempfile.TemporaryDirectory() as directory:
            result = run_repeated(config, factory, output_dir=directory)
            self.assertEqual(result['repetitions_completed'], 5)
            self.assertEqual(calls, config['experiment_seeds'])
            self.assertEqual(result['aggregate']['RMSE']['runs'], 5)
        summary = summarize_runs([{'MAE': 1.}, {'MAE': 3.}])
        self.assertAlmostEqual(summary['MAE']['std'], math.sqrt(2))

    def test_gaussian_metrics_known_case(self):
        result = forecast_metrics(tf.zeros([2]), tf.ones([2]), tf.zeros([2]))
        self.assertAlmostEqual(result['NLL'], 0.5 * math.log(2 * math.pi), places=6)
        self.assertAlmostEqual(result['CRPS'], (math.sqrt(2) - 1) / math.sqrt(math.pi), places=6)
        self.assertEqual(result['PICP@95'], 1.0)
        self.assertIsNone(result['MAPE_percent'])

    def test_regime_boundaries_and_72_differences(self):
        generation = tf.tile(tf.range(73, dtype=tf.float32)[None, :], [2, 1])
        np.testing.assert_allclose(historical_volatility(generation, tf.zeros_like(generation)), 1.0)
        boundaries = fit_regime_boundaries(tf.constant([0., 1., 2., 3.]))
        np.testing.assert_allclose(boundaries, [1., 2.])
        scores = regime_metrics(tf.zeros([3, 1, 1]), tf.ones([3, 1, 1]), tf.zeros([3, 1, 1]), [0.5, 1.5, 2.5], boundaries)
        self.assertEqual([scores[name]['samples'] for name in ('low', 'moderate', 'high')], [1, 1, 1])

    def test_invalid_target_and_tensor_shapes(self):
        config = self.compact()
        config['input_columns'].append('balance')
        with self.assertRaises(ValueError):
            MDHGNet(config)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            gaussian_nll(tf.zeros([2]), tf.ones([2]), tf.zeros([2, 1]))


if __name__ == '__main__':
    unittest.main()
