import tensorflow as tf
keras = tf.keras
tfkl = keras.layers
K = keras.backend

import math


class ConvFreqAnalysis(tfkl.Layer):
    """
    decompose every input across time into a frequency at the moment
    aka: differentiator
    """

    def __init__(self,
                 periods,
                 window_size,
                 **kwargs):
        super(ConvFreqAnalysis, self).__init__(**kwargs)

        self.periods = periods
        self.window_size = window_size

        self.period_filters = tf.range(window_size)[:, None] \
                              * tf.range(len(self.periods))[None, :]
        # [N, T]; this is different than in `FullFreqAnalysis`
        self.period_dotprods = K.sum(self.period_filters * self.period_filters, axis=1)


    def call(self, inputs):
        # [B, T, X]
        inputs_transposed = K.transpose(inputs, (0,2,1))
        # [B, X, T]
        centered_unnormalized_scores = K.conv1d(
            x=inputs_transposed,
            kernel=self.period_filters, # I don't know if this accept multiple filters
            strides=1,
            padding="same"
        )
        aligned_unnormalized_scores = tf.roll(
            input=centered_unnormalized_scores,
            shift=self.window_size / 2 - 1,
            axis=-2,
            name=f"{self.name}_aligned_unnormalized_scores")
        # [B, X, T, N]
        return aligned_unnormalized_scores / self.period_dotprods


class FullFreqAnalysis(tfkl.Layer):
    """
    decompose every input across time into a frequency at the moment
    aka: differentiator
    """

    def __init__(self,
                 periods,
                 **kwargs):
        super(FullFreqAnalysis, self).__init__(**kwargs)

        self.periods = periods

    def build(self, input_shape):
        # X: [B, T, X]
        Ts = tf.range(input_shape[-2])
        self.bases = tf.sin(2 * math.pi * Ts[None, :] / self.periods[:, None])
        self.bases_dotprod = K.sum(self.bases * self.bases, axis=1)

    def call(self, inputs):
        """
        This approach is designed to take up the entire time window
        This is NOT the same as 1D convolution filters over time
        """
        inputs_perm = K.permute_dimensions(inputs, [0,2,1])[...,None] # [B, X, T, 1]
        convoluted = self.bases * inputs_perm # [B, X, T, N]
        components = K.sum(convoluted, axis=-2) # [B, X, N]
        return components / self.bases_dotprod # [B, X, N]
