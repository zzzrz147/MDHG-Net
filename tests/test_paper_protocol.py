from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import TensorDataset, DataLoader

from mdhg_net import MDHGNet
from mdhg_net.paper_protocol import audit_config, validate_paper_settings
from mdhg_net.experiments import experiment_configs, run_repeated, summarize_runs
from mdhg_net.training import fit, make_loaders, load_model
from mdhg_net.modules.hypergraph_convolution import HypergraphConvolution
from mdhg_net.inference import mc_predict
from mdhg_net.evaluation import historical_volatility, fit_regime_boundaries, regime_metrics, forecast_metrics


class PaperProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.default = json.loads((Path(__file__).parents[1]/'configs/default.json').read_text(encoding='utf-8'))

    def compact(self):
        config = copy.deepcopy(self.default)
        config.update(seq_len=12, temporal_anchor_len=12, window_size=[1,4,8], d_model=8, gru_hidden=8,
                      train_epochs=1, patience=1, batch_size=2, mc_passes=2, enforce_paper_settings=False)
        return config

    def test_default_reported_settings_and_explicit_unknowns(self):
        validate_paper_settings(self.default)
        rows = audit_config(self.default)
        self.assertFalse([row for row in rows if row['status']=='differs'])
        statuses = {row['setting']:row['status'] for row in rows}
        self.assertEqual(statuses['encoder_stride'], 'implementation_choice')
        self.assertEqual(statuses['input_variable_count'], 'user_override')
        self.assertEqual(statuses['sample_window_stride'], 'external_contract')

    def test_paper_mode_blocks_changed_training_settings(self):
        for name, value in [('learning_rate',0.01),('train_epochs',20),('patience',3),('batch_size',32),('mc_passes',5)]:
            config = copy.deepcopy(self.default); config[name]=value
            with self.subTest(name=name), self.assertRaises(ValueError):
                MDHGNet(config)

    def test_actual_loader_batch_size_is_checked(self):
        config = self.compact()
        x,y = torch.randn(4,12,24),torch.randn(4,1,1)
        wrong = DataLoader(TensorDataset(x,y),batch_size=4)
        with self.assertRaisesRegex(ValueError, 'batch_size'):
            fit(MDHGNet(config),wrong,wrong,config)

    def test_loader_factory_applies_batch_and_preserves_evaluation_samples(self):
        config = self.compact(); config['batch_size']=3
        dataset = TensorDataset(torch.randn(7,12,24),torch.randn(7,1,1))
        train,val,test = make_loaders(dataset,dataset,dataset,config)
        self.assertEqual(train.batch_size,3)
        self.assertEqual(sum(y.shape[0] for _,y in val),7)
        self.assertEqual(sum(y.shape[0] for _,y in test),7)

    def test_mc_default_comes_from_the_model(self):
        config = self.compact(); config['mc_passes']=3; config['confidence_levels']=[0.8]
        model = MDHGNet(config)
        with patch.object(model,'forward',wraps=model.forward) as forward:
            prediction = mc_predict(model,torch.randn(2,12,24))
        self.assertEqual(forward.call_count,3)
        self.assertEqual(set(prediction['intervals']),{0.8})

    def test_matrix_form_of_equation_23(self):
        module = HypergraphConvolution(2,2,0.0)
        h = torch.tensor([[0.4,0.2],[0.1,0.6]])
        x = torch.tensor([[1.,3.],[2.,4.]])
        wh = torch.tensor([[1.,0.25],[0.5,1.]])
        with torch.no_grad():
            module.node_proj.weight.copy_(torch.eye(2)); module.edge_weight.copy_(wh)
        result,_ = module(x[None,None],h[None,None])
        dv = torch.diag(h.sum(dim=1).pow(-0.5)); de = torch.diag(h.sum(dim=0).pow(-1))
        expected = torch.relu(dv @ h @ wh @ de @ h.T @ dv @ x)
        torch.testing.assert_close(result[0,0],expected)
        result.sum().backward()
        self.assertGreater(module.edge_weight.grad[0,1].abs().item(),0.)

    def test_early_stopping_restores_best_before_maximum_epochs(self):
        config=self.compact(); config.update(train_epochs=8,patience=2)
        dataset=TensorDataset(torch.randn(4,12,24),torch.randn(4,1,1))
        train,val,_=make_loaders(dataset,dataset,dataset,config)
        losses=[2.,5.,2.,4.,2.,4.,2.,4.]
        with patch('mdhg_net.training.trainer.run_epoch',side_effect=losses):
            summary=fit(MDHGNet(config),train,val,config)
        self.assertEqual(summary['best_epoch'],2)
        self.assertEqual(summary['epochs_completed'],4)
        self.assertEqual(summary['effective_settings']['early_stop_monitor'],'val_nll')
        self.assertTrue(all(row['learning_rate']==0.001 for row in summary['history']))

    def test_separate_lead_and_scale_experiment_plans(self):
        plans=experiment_configs(self.default,'horizons')
        self.assertEqual([case['forecast_horizons'] for case in plans.values()],[[12],[24],[48],[72]])
        scales=experiment_configs(self.default,'scales')
        self.assertEqual([len(case['window_size']) for case in scales.values()],[2,3,4,5])
        self.assertTrue(list(scales.values())[-1]['implementation_notes'])
        self.assertEqual(len(experiment_configs(self.default,'ablations')),6)

    def test_five_seed_training_and_post_training_evaluation(self):
        config=self.compact()
        calls=[]
        def factory(cfg,seed):
            calls.append(seed)
            generator=torch.Generator().manual_seed(123)
            dataset=TensorDataset(torch.randn(4,12,24,generator=generator),torch.randn(4,1,1,generator=generator))
            return make_loaders(dataset,dataset,dataset,cfg)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            result=run_repeated(config,factory,output_dir=directory)
            self.assertEqual(result['repetitions_completed'],5)
            self.assertEqual(calls,config['experiment_seeds'])
            self.assertEqual(result['aggregate']['RMSE']['runs'],5)
            self.assertEqual(result['aggregate']['RMSE']['std'] >= 0,True)
            self.assertTrue((Path(directory)/'summary.json').exists())
            for run in result['runs']:
                self.assertEqual(run['training']['epochs_completed'],1)
                self.assertEqual(run['training']['effective_settings']['train_loader']['batch_size'],2)

    def test_aggregation_of_seed_results_uses_sample_standard_deviation(self):
        result=summarize_runs([{'MAE':1.0},{'MAE':3.0}])
        self.assertEqual(result['MAE']['mean'],2.0)
        self.assertAlmostEqual(result['MAE']['std'],2**0.5)

    def test_historical_volatility_uses_72_differences_and_fixed_training_quantiles(self):
        g=torch.arange(73,dtype=torch.float32)[None,:].repeat(2,1)
        l=torch.zeros_like(g)
        torch.testing.assert_close(historical_volatility(g,l),torch.ones(2))
        with self.assertRaises(ValueError):
            historical_volatility(g[:,:72],l[:,:72])
        boundaries=fit_regime_boundaries(torch.tensor([0.,1.,2.,3.]))
        torch.testing.assert_close(boundaries,torch.tensor([1.,2.],dtype=torch.float64))
        score=regime_metrics(torch.zeros(3,1,1),torch.ones(3,1,1),torch.zeros(3,1,1),
                             torch.tensor([0.5,1.5,2.5]),boundaries)
        self.assertEqual([score[name]['samples'] for name in ('low','moderate','high')],[1,1,1])

    def test_alarm_threshold_probability_is_applied(self):
        mu,var,target=torch.tensor([0.]),torch.tensor([1.]),torch.tensor([0.])
        low=forecast_metrics(mu,var,target,thresholds=[0.0],alarm_probability=0.5)
        high=forecast_metrics(mu,var,target,thresholds=[0.0],alarm_probability=0.6)
        self.assertEqual(low['low_balance']['0.0']['true_positive'],1)
        self.assertEqual(high['low_balance']['0.0']['true_positive'],0)


if __name__=='__main__':
    unittest.main()
