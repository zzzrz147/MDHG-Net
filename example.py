import json
from pathlib import Path
import tensorflow as tf
from mdhg_net import MDHGNet
from mdhg_net.training import AdamW, set_seed
from mdhg_net.losses import gaussian_nll
from mdhg_net.inference import mc_predict


def main():
    config = json.loads((Path(__file__).parent / 'configs/default.json').read_text(encoding='utf-8'))
    set_seed(config['seed'])
    model = MDHGNet(config)
    x = tf.random.normal([2, config['seq_len'], len(config['input_columns'])])
    y = tf.random.normal([2, len(config['forecast_horizons']), 1])
    betas = config['optimizer_betas']
    optimizer = AdamW(config['learning_rate'], config['weight_decay'], betas[0], betas[1], config['optimizer_eps'])
    with tf.GradientTape() as tape:
        output = model(x, training=True)
        loss = gaussian_nll(output['mu'], output['variance'], y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients([(gradient, variable) for gradient, variable in zip(gradients, model.trainable_variables)
                              if gradient is not None])
    prediction = mc_predict(model, x)
    print('TensorFlow:', tf.__version__)
    print('Input:', x.shape, 'Output:', prediction['mu'].shape)
    print('Synthetic NLL:', float(loss.numpy()))
    print('Mean epistemic variance:', float(tf.reduce_mean(prediction['epistemic_variance']).numpy()))


if __name__ == '__main__':
    main()
