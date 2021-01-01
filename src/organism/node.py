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

        self._fe = 0
        self._rolling_fe = 0

        self.children = list()
        self._parents = dict()
        self._target_states = dict()
        self._state = {
            'controllable': { },
            'uncontrollable': {
                'fe': self._get_fe
            }
        }

    @property
    def get_state(self):
        """returns a dict like:
        { 'controllable': dict<str, any>,
          'uncontrollable': dict<str, any> }
        """
        return self._state

    def set_target_state(self, target_state, callee):
        """
        just sets the 'controllable' part of state
        """
        self._target_states[callee] = target_state

    def bottom_up(self):
        """perception. update self._state"""
        raise NotImplementedError()

    def top_down(self):
        """action. act according to self_target_states"""
        self._target_states = dict()

        if self._rolling_fe > self._adaptation_threshold and self._rolling_fe > 0:
            self._maybe_apply_adaptation()
            self._start_adaptation()
            self._adaptive_potential -= self._rolling_fe

    def _start_adaptation(self):
        """simple models may adapt in realtime instead of using this dedicated fn"""
        pass

    def _set_fe(self, fe):
        self._fe = fe
        self._rolling_fe = self._adaptive_beta * self._rolling_fe + \
                           (1-self._adaptive_beta) * self._fe

    @property
    def _get_fe(self):
        return self._fe

    @property
    def get_state_space(self):
        """
        state space of node

        Physical or information-theoretic 'fe' should be
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
                'fe': (1,),
                ...
        }
        """
        def parse_dict(d):
            d_ret = dict()
            for k,v in d.items():
                if isinstance(v, dict):
                    d_ret[k] = parse_dict(v)
                else:
                    d_ret[k] = K.int_shape(v)
            return d_ret

        return parse_dict(self._state)