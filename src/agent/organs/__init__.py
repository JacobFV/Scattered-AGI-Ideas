class Organ:

    def __init__(self, name):
        self._name = name

    def step(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def get_observation_space(self):
        raise NotImplementedError()

    def get_action_space(self):
        raise NotImplementedError()

    def get_observation(self):
        raise NotImplementedError()

    def do_action(self):
        raise NotImplementedError()

    @property
    def unity_primitive(self):
        raise NotImplementedError()

    @property
    def get_name(self):
        return self._name