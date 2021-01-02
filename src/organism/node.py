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

        self._target_states = dict()

    @property
    def get_info_state(self):
        """returns a dict like:
        { 'controllable': dict<str, val|dict>,
          'uncontrollable': dict<str, val|dict> }
        """
        return {
            'controllable': { },
            'uncontrollable': {
                'free_energy': self.get_free_energy
            }
        }

    def set_target_info_state(self, target_controllable_state, weight, callee):
        """
        just sets the 'controllable' part of state
        """
        self._target_states[callee] = [weight, target_controllable_state]

    def bottom_up(self):
        """perception. update self._info_state"""
        raise NotImplementedError()

    def top_down(self):
        """action. act according to self_target_states"""
        self._target_states = dict()

        if self._rolling_free_energy > self._adaptation_threshold and self._rolling_free_energy > 0:
            self._start_adaptation()
            self._adaptive_potential -= self._rolling_free_energy

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

    @property
    def get_info_state_space(self):
        """
        state space of node

        Physical or information-theoretic 'free_energy' should be
        a key in every 'uncontrollable' state space. In the
        InformationNodes, it refers to the KL divergence of
        state from previous predictions. In the body Organ's,
        it is defined by a heterogenous overlapping set of metrics
        such as distance to zero (energy node), rolling beta
        average distance from target orientation (joints), or
        nervous signal pulse rate (muscles). This metric should
        always be in [0,inf)

        should return a dict like:
        {
            'controllable': dict<str, any>,
                ...
            'uncontrollable': dict<str, any>
                'free_energy': (1,),
                ...
        }
        """
        return utils.structured_op(self.get_info_state, lambda v: K.int_shape(v))