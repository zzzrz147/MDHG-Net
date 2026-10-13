import tensorflow as tf


class TemporalDecoder(tf.keras.layers.Layer):
    def __init__(self, channels, hidden, dropout, num_layers=1, gru_dropout=0.0):
        super().__init__()
        self.gru_layers = [tf.keras.layers.GRU(hidden, activation='tanh', recurrent_activation='sigmoid',
                                              reset_after=True, return_sequences=index < num_layers - 1)
                           for index in range(num_layers)]
        self.inter_layer_dropouts = [tf.keras.layers.Dropout(gru_dropout) for _ in range(max(0, num_layers - 1))]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, sequence, training=False, dropout_training=None):
        output = sequence
        for index, gru in enumerate(self.gru_layers):
            output = gru(output, training=training)
            if index < len(self.inter_layer_dropouts):
                output = self.inter_layer_dropouts[index](output, training=training)
        return self.dropout(output, training=training if dropout_training is None else dropout_training)
