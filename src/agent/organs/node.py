from . import utils

import tensorflow as tf
keras = tf.keras
tfkl = keras.layers
K = keras.backend


class Node(utils.Freezable, utils.PermanentName):

    def __init__(self,
                 name,
                 adaptive_potential,
                 adaptation_beta=0.99,
                 adaptation_threshold=10):

        super(Node, self).__init__(name=name)

        self._adaptive_potential = adaptive_potential
        self._adaptive_beta = adaptation_beta
        self._adaptation_threshold = adaptation_threshold

        self._free_energy = 0
        self._rolling_free_energy = 0

    def bottom_up(self):
        """perception. update self._info_state"""
        raise NotImplementedError()

    def update_state(self):
        """perform logic updates in the middle of environment step"""
        if self._rolling_free_energy > self._adaptation_threshold and self._rolling_free_energy > 0:
            self._start_adaptation()
            self._adaptive_potential -= self._rolling_free_energy

    def top_down(self):
        """action. act according to self_target_states"""
        raise NotImplementedError()

    def _start_adaptation(self):
        """simple models may adapt in realtime instead of using this dedicated fn"""
        pass

    def set_free_energy(self, free_energy):
        self._free_energy = free_energy
        self._rolling_free_energy = self._adaptive_beta * self._rolling_free_energy + \
                                    (1-self._adaptive_beta) * self._free_energy

    @property
    def get_free_energy(self):
        """
        I keep this as a getr to
        1) encourage use of the `_set_fe` function
        2) allow a specific case of dynamic free_energy computation in `EnergyNode`.
        """
        return self._free_energy
