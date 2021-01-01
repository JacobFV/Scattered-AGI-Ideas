class Freezable:
    """
    Freezables must first be initialized before unfreezing
    """

    def freeze(self, freeze_to_path):
        raise NotImplementedError()

    def unfreeze(self, unfreeze_from_path):
        raise NotImplementedError()


class SimulationEnvCommunicator:

    def add_to_env_simulation(self):
        raise NotImplementedError()

    def remove_from_env_simulation(self):
        raise NotImplementedError()


class PermanentName:

    def __init__(self, name):
        self._name = name

    @property
    def get_name(self):
        return self._name