from . import NodeOrgan, EdgeOrgan

import tensorflow as tf
keras = tf.keras
tfkl = keras.layers
K = keras.backend


class EnergyNode(NodeOrgan):

    def __init__(self, name, d_energy=8):

        super(EnergyNode, self).__init__(name=name, adaptive_potential=0)

        self._energy = K.zeros(shape=(d_energy,))

    def top_down(self):
        self._set_fe(K.sum(self.get_energy**0.5)) # this is arbitrary

    @property
    def get_energy(self):
        return self._energy

    def set_energy(self, energy):
        tf.assert_greater(energy, 0, 'energy components must be non-negative')
        self._energy = energy