# 论文参数与代码逐项对照

依据本次提供的 manuscript.docx 第 3 节、表 2 和第 4 节实验描述，对版本 0.3.0 的参数与实际执行逻辑进行核对。

文稿明确给出的主设置已落实为：AdamW、学习率 0.001、weight decay 0.001、批大小 64、最多 200 轮、早停耐心 20、dropout 0.3、历史长度与时间锚点 72、默认尺度 [1,12,24]、稀疏阈值 [0.05,0.10,0.15]、MC 推理 50 次，以及每个实验配置五个随机种子。

完整代码仍按模块分文件，不包含公开数据、CSV 读取或数据预处理。实际数据的时间划分、采样窗口与标签由已有 Dataset 提供。

## 执行中的检查

- 主配置默认启用 `enforce_paper_settings=true`，拒绝与文稿报告设置不同的主参数；自定义试跑需显式关闭。
- `fit()` 检查真实 DataLoader 批大小与模型配置，实际构建 AdamW，并逐轮记录有效学习率。
- 只优化 Gaussian NLL，以验证 NLL 早停，恢复最佳权重。200 轮是上限。
- `run_repeated()` 真正执行五次独立种子训练，在每次训练结束后评估测试集并汇总均值与标准差。
- `mc_predict()` 与 `predict_loader()` 默认从模型读取 MC 次数；主配置为 50。
- 12/24/48/72 小时提前量分别创建并训练单一输出模型。
- 式 (23) 的 Wh 按 Ms×Ms 完整矩阵实现；旧版本对角权重可迁移为矩阵对角元素。

## 步长与次数的含义

预测提前量 h 是 [12,24,48,72] 小时；默认任务仅预测第 12 小时的一个值。尺度窗口 [1,12,24] 是各自独立的历史聚合块长度，各尺度内部聚合按窗口长度步进。卷积 stride=1 是保留时间分辨率的明确实现选择，文稿未报告其具体值。用于创建训练样本的滑窗步长也未写明，且属于外部 Dataset；不能将它与预测提前量或卷积 stride 混用。

## 已报告参数与外部接口约定

| 设置 | 当前配置/实现 | 文稿要求 | 状态 | 来源 |
| --- | --- | --- | --- | --- |
| sampling_interval_hours | 1.0 | 1.0 | 文稿明确，设置相符 | Eq. (6), Table 2 |
| seq_len | 72 | 72 | 文稿明确，设置相符 | Section 4.1, Table 2 |
| temporal_anchor_len | 72 | 72 | 文稿明确，设置相符 | Section 4.1, Table 2 |
| d_model | 64 | 64 | 文稿明确，设置相符 | Section 4.1, Table 2 |
| dropout | 0.3 | 0.3 | 文稿明确，设置相符 | Section 4.1, Table 2 |
| optimizer | AdamW | AdamW | 文稿明确，设置相符 | Section 4.1, Table 2 |
| learning_rate | 0.001 | 0.001 | 文稿明确，设置相符 | Section 4.1, Table 2 |
| weight_decay | 0.001 | 0.001 | 文稿明确，设置相符 | Section 4.1, Table 2 |
| batch_size | 64 | 64 | 文稿明确，设置相符 | Section 4.1, Table 2 |
| train_epochs | 200 | 200 | 文稿明确，设置相符 | Section 4.1, Table 2; maximum, not mandatory completion |
| patience | 20 | 20 | 文稿明确，设置相符 | Section 4.1, Table 2 |
| loss | gaussian_nll | gaussian_nll | 文稿明确，设置相符 | Eqs. (43)-(44), Table 2 |
| mc_passes | 50 | 50 | 文稿明确，设置相符 | Section 4.1, Table 2 |
| repetitions | 5 | 5 | 文稿明确，设置相符 | Section 4.1 |
| train_ratio | 0.8 | 0.8 | 由外部 Dataset 履行 | Table 2; applied by external dataset split |
| val_ratio | 0.1 | 0.1 | 由外部 Dataset 履行 | Table 2; applied by external dataset split |
| test_ratio | 0.1 | 0.1 | 由外部 Dataset 履行 | Table 2; applied by external dataset split |
| confidence_levels | [0.9, 0.95] | [0.9, 0.95] | 文稿明确，设置相符 | Section 4.1, Table 4 |
| low_balance_thresholds | [500.0, 600.0, 700.0] | [500.0, 600.0, 700.0] | 文稿明确，设置相符 | Section 4.4, Table 7 |
| low_balance_reference | 600.0 | 600.0 | 文稿明确，设置相符 | Section 4.4 |
| alarm_probability | 0.5 | 0.5 | 文稿明确，设置相符 | Section 4.4 |
| volatility_history_hours | 72 | 72 | 文稿明确，设置相符 | Section 4.5, Eq. (45) |
| volatility_quantiles | [0.3333333333333333, 0.6666666666666666] | [0.3333333333333333, 0.6666666666666666] | 文稿明确，设置相符 | Section 4.5 |
| forecast_horizons | [12] | [12, 24, 48, 72] | 文稿明确，设置相符 | Sections 4.1 and 4.3; train each lead separately |
| window_size | [1, 12, 24] | [[1, 12], [1, 12, 24], [1, 12, 24, 48], [1, 12, 24, 48, 72]] | 文稿明确，设置相符 | Sections 4.1 and 4.3 |
| incidence_thresholds | [0.05, 0.1, 0.15] | [0.05, 0.1, 0.15] | 文稿明确，设置相符 | Section 4.1, default three-scale setup |
| input_variable_count | 24 | 23 | 保留用户确认的输入列 | Section 3.1, Table 2; provided-column contract retained |
| sample_window_stride | external Dataset | 未报告具体值 | 由外部 Dataset 履行 | No sliding-sample stride is specified; dataset creation is outside this package |
| hyperedge_weight_matrix | full learnable Ms x Ms, identity initialization | Wh in R^(Ms x Ms) | 文稿明确，设置相符 | Eq. (23); initialization is an implementation choice |

## 文稿没有完整给出的细节

以下数值可运行且已显式保存或注明，但文稿不足以确定它们就是原实验中的具体取值。严格复现这些部分需要作者的原始配置。

| 设置 | 当前配置/实现 | 未报告的内容 |
| --- | --- | --- |
| encoder_kernel_size | 3 | Convolution kernel size is not stated. |
| encoder_stride | 1 | Convolution stride is not stated; 1 retains the finest resolution. |
| encoder_layers | 2 | The number of convolution layers is not stated. |
| encoder_activation | relu | The nonlinear activation delta is not named. |
| interpolation_align_corners | false | Linear interpolation is stated; align_corners is not. |
| gru_hidden | 64 | A separate GRU hidden size is not stated; it is set to the latent size. |
| gru_layers | 1 | The number of GRU layers is not stated. |
| gru_dropout | 0.0 | Internal GRU dropout is not separately stated. |
| scale_se_reduction | 4 | Excitation bottleneck size is not stated. |
| hyper_num | [6, 6, 6] | The number of candidate hyperedges at each scale is not stated. |
| variance_epsilon | 1e-06 | Epsilon is positive; its numerical value is not stated. |
| optimizer_betas | [0.9, 0.999] | AdamW beta values are not stated; standard defaults are used. |
| optimizer_eps | 1e-08 | AdamW epsilon is not stated; the standard default is used. |
| lr_scheduler | none | Only initial learning rate is stated; no scheduler is specified. |
| early_stop_min_delta | 0.0 | Early-stopping minimum improvement is not stated. |
| grad_clip | 未报告具体值 | Gradient clipping is not stated and is disabled by default. |
| num_workers | 0 | DataLoader worker count is not stated. |
| pin_memory | false | DataLoader pin_memory is not stated. |
| train_drop_last | false | Handling of the last incomplete batch is not stated. |
| experiment_seeds | [2025, 2026, 2027, 2028, 2029] | Five seeds are required; their numerical values are not stated. |
| normalization | scale LayerNorm or caller-supplied training statistics | Eq. (7) |
| normalization_epsilon | 1e-05 | LayerNorm standard default; the manuscript does not specify its epsilon |
| zero_padding_side | oldest end, left | Eq. (4) defines padded length but not padding direction; no padding occurs in the default scale set |
| parameter_initialization | PyTorch layer defaults, Xavier-uniform edge prototypes | Initialization distributions are not reported |
| hyperedge_weight_initialization | identity | Eq. (23) does not give initialization values |
| probabilistic_score_distribution | moment-matched Gaussian | Eq. (42) states Gaussian interval approximation; MC-mixture score details are not specified |
| MAPE_near_zero_handling | exclude abs(target)<=1e-6 | MAPE is supplementary; the exact denominator treatment is not specified |
| PINAW_normalizer | evaluated target maximum minus minimum | The exact PINAW normalization definition is not reported |
| reported_seed_standard_deviation | sample standard deviation, ddof=1 | Mean and standard deviation are reported, but ddof is not specified |

## 数据列与非默认尺度

按前述确认保留现有 24 个非目标列：17 个功率变量和 7 个气象变量。文稿写 23 个变量、18+5，本包不自行编造缺失的市场变量或删除已确认列。此项属于已知输入约定差异。

文稿没有逐项给出 2/4/5 尺度实验的阈值。实验生成器默认在已有阈值后重复 0.15，并在非三尺度计划中记录说明；可通过 `thresholds_by_scale_count` 传入作者的对应数值。候选超边数仍为每尺度 6，文稿未报告该数值。

0.8/0.1/0.1 记录为时间划分约定。没有原始时间轴的模型代码不能验证外部 Dataset 实际是否遵守比例、样本滑窗步长或标签提前量；能够检查 Dataset 主动暴露的 `feature_names` 与 `forecast_horizons`。

## 验证与使用

25 项本地测试、7 组旧版生成包权重迁移与预测对照、50 次随机张量 MC 推理和 wheel 构建通过。真实数据上的五种子完整训练未在本次执行。

在仓库根目录运行 `python run_experiments.py --suite horizons --describe` 可查看当前实验计划。配置文件为 `configs/default.json`，机器可读对照为 `docs/paper_alignment.json`。

源文稿 SHA256：`2ef0d6da41b439e2d5eb966f1406e8daaa403b5f1094bef113669186a72314c2`。
