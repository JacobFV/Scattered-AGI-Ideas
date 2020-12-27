class Freezable:
    """
    Freezables must first be initialized before unfreezing
    """

    def __init__(self, freezables=[]):
        self.freezables = freezables

    def freeze(self, freeze_to_path):
        for freezable in self.freezables:
            freezable.freeze(freeze_to_path)

    def unfreeze(self, unfreeze_from_path):
        for freezable in self.freezables:
            freezable.unfreeze(unfreeze_from_path)


class Stepable:

    def __init__(self, stepables=[]):
        self.stepables = stepables

    def step(self):
        for stepable in self.stepables:
            stepable.step()


class Trainable:

    def __init__(self, trainables=[]):
        self.trainables = trainables

    def train(self):
        for trainable in self.trainables:
            trainable.train()


class SimulationEnvCommunicator:

    def __init__(self, simulation_env_communicators=[]):
        self.simulation_env_communicators = simulation_env_communicators

    def add_to_env_simulation(self):
        for simulation_env_communicator in self.simulation_env_communicators:
            simulation_env_communicator.add_to_env_simulation()

    def remove_from_env_simulation(self):
        for simulation_env_communicator in self.simulation_env_communicators:
            simulation_env_communicator.remove_from_env_simulation()


class PermanentName:

    def __init__(self, name):
        self._name = name

    @property
    def get_name(self):
        return self._name