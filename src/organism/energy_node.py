from . import utils

class EnergyNode(utils.Freezable, utils.PermanentName):

    def __init__(self, name, adaptive_potential, adaptation_threshold=10):
        super(EnergyNode, self).__init__(name=name)

        self.children = list()
        self._parents = dict()
        self._target_states = dict()
        self._state = None

        self.adaptive_potential = adaptive_potential
        self._fe = 0
        self._rolling_fe = 0
        self._adaptation_threshold = adaptation_threshold

    @property
    def get_state(self):
        """should return a dict like:
        {
            'controllable': dict<str, any>,
            'uncontrollable': dict<str, any>
        }
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

        self._check_for_new_adaptations()
        if self._rolling_fe > self._adaptation_threshold:
            self._adapt()

    def _check_for_new_adaptations(self):
        """simple models may adapt in realtime instead of using this dedicated fn"""
        raise NotImplementedError()

    def _adapt(self):
        """simple models may adapt in realtime instead of using this dedicated fn"""
        raise NotImplementedError()

    def _set_fe(self, fe):
        self._fe = fe
        self._rolling_fe = # TODO

    # TODO erase _fe from InformationNode and use these get and setrs instead

    def _get_fe(self):
        return self._fe

    def

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
                'fe': scalar,
                ...
        }
        """
        raise NotImplementedError()