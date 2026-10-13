import tensorflow as tf
from .hypergraph_construction import HypergraphConstruction
from .hypergraph_convolution import HypergraphConvolution


class DynamicHypergraph(tf.keras.layers.Layer):
    def __init__(self, channels, prior, threshold, dropout, dynamic=True):
        super().__init__()
        self.construction = HypergraphConstruction(channels, prior, threshold, dynamic)
        self.convolution = HypergraphConvolution(channels, prior.shape[1], dropout)

    def call(self, states, descriptor, training=False):
        incidence, details = self.construction(states, descriptor)
        local, edges = self.convolution(states, incidence, training=training)
        return local, edges, incidence, details
