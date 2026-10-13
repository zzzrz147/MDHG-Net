import tensorflow as tf


class TemporalEncoder(tf.keras.layers.Layer):
    def __init__(self, channels, dropout, kernel_size=3, stride=1, num_layers=2,
                 activation='relu', align_corners=False):
        super().__init__()
        if activation != 'relu':
            raise ValueError('encoder_activation must be relu.')
        self.channels = channels
        self.align_corners = bool(align_corners)
        self.convolutions = [tf.keras.layers.Conv1D(channels, kernel_size, strides=stride, padding='valid')
                             for _ in range(num_layers)]
        self.paddings = [tf.keras.layers.ZeroPadding1D(kernel_size // 2) for _ in range(num_layers)]
        self.dropouts = [tf.keras.layers.Dropout(dropout) for _ in range(num_layers)]

    def call(self, x, anchor_len, training=False):
        shape = tf.shape(x)
        encoded = tf.reshape(tf.transpose(x, [0, 2, 1]), [shape[0] * shape[2], shape[1], 1])
        for padding, convolution, dropout in zip(self.paddings, self.convolutions, self.dropouts):
            encoded = dropout(tf.nn.relu(convolution(padding(encoded))), training=training)
        encoded = tf.raw_ops.ResizeBilinear(images=encoded[:, :, None, :], size=[anchor_len, 1],
                                            align_corners=self.align_corners,
                                            half_pixel_centers=not self.align_corners)[:, :, 0, :]
        return tf.transpose(tf.reshape(encoded, [shape[0], shape[2], anchor_len, self.channels]), [0, 2, 1, 3])
