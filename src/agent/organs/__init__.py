class Organ:

    def __init__(self, name, agent):
        self._name = name
        self.agent = agent

    def step(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def save_frozen_copy(self):
        raise NotImplementedError()

    @staticmethod
    def init_from_freeze(dir):
        pass

    def get_observation_space(self):
        raise NotImplementedError()

    def get_action_space(self):
        raise NotImplementedError()

    def get_observation(self):
        raise NotImplementedError()

    def do_action(self):
        raise NotImplementedError()

    @property
    def unity3d_primitive(self):
        raise NotImplementedError()

    @property
    def get_name(self):
        return self._name