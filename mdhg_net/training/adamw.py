import tensorflow as tf


class AdamW(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, weight_decay=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, name='AdamW', **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('weight_decay', weight_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon

    def _create_slots(self, var_list):
        for variable in var_list:
            self.add_slot(variable, 'm')
        for variable in var_list:
            self.add_slot(variable, 'v')

    def _resource_apply_dense(self, grad, variable, apply_state=None):
        dtype = variable.dtype.base_dtype
        lr = self._decayed_lr(dtype)
        beta_1 = self._get_hyper('beta_1', dtype)
        beta_2 = self._get_hyper('beta_2', dtype)
        decay = self._get_hyper('weight_decay', dtype)
        step = tf.cast(self.iterations + 1, dtype)
        first, second = self.get_slot(variable, 'm'), self.get_slot(variable, 'v')
        first_update = first.assign(beta_1 * first + (1.0 - beta_1) * grad)
        second_update = second.assign(beta_2 * second + (1.0 - beta_2) * tf.square(grad))
        with tf.control_dependencies([first_update, second_update]):
            first_hat = first_update / (1.0 - tf.pow(beta_1, step))
            second_hat = second_update / (1.0 - tf.pow(beta_2, step))
            update = variable.assign(variable * (1.0 - lr * decay) - lr * first_hat / (tf.sqrt(second_hat) + self.epsilon))
        return tf.group(first_update, second_update, update)

    def _resource_apply_sparse(self, grad, variable, indices, apply_state=None):
        dense = tf.scatter_nd(tf.expand_dims(indices, 1), grad, tf.shape(variable))
        return self._resource_apply_dense(dense, variable, apply_state)

    def get_config(self):
        result = super().get_config()
        result.update(learning_rate=self._serialize_hyperparameter('learning_rate'),
                      weight_decay=self._serialize_hyperparameter('weight_decay'),
                      beta_1=self._serialize_hyperparameter('beta_1'), beta_2=self._serialize_hyperparameter('beta_2'),
                      epsilon=self.epsilon)
        return result
