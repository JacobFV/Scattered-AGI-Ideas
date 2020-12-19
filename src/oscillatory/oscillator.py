import tensorflow as tf
keras = tf.keras
tfkl = keras.layers
K = keras.backend


class OscillatorCell(tfkl.AbstractRNNCell):

    def __init__(self,
                 units,
                 periods,
                 rate_bias=1.0,
                 **kwargs):
        """
        rate bias (Tensor-like): either a scalor or vector with
            `len(periods)` elements to individual evolve time
        """
        super(OscillatorCell, self).__init__(**kwargs)

        self.units = units
        self.periods = periods
        self.rate_bias = rate_bias

    @property
    def state_size(self):
        return self.periods.shape[0]

    def build(self, input_shape):
        self.W_in = self.add_weight(name=f'{self.name}_W_in',
                                    shape=(input_shape[-1], self.periods.shape[0]),
                                    initializer='glorot')
        self.W_out = self.add_weight(name=f'{self.name}_W_out',
                                     shape=(self.periods.shape[0], self.units),
                                     initializer='glorot')

    def call(self, inputs, states):
        new_states = states + keras.activations.relu(inputs) @ self.W_in + self.rate_bias
        outputs = K.sin(states[...,None] * self.periods) @ self.W_out


def Oscillator(**kwargs):
    """
    generate waveform ove rtime
    aka: integrator
    """
    return tfkl.RNN(OscillatorCell(**kwargs))