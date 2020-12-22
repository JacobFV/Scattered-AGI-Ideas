class Organ:

    def __init__(self, name):
        self.name = name

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


class EnergyVessel(Organ): pass
class EnergyPump(Organ): pass
class EnergyConsumingOrgan(Organ):

    def __init__(self, name, consumption_range=[0,100], energy_cost=0.1):
        super(EnergyConsumingOrgan, self).__init__(name=name)
        self.consumption_range = consumption_range
        self.energy_cost = energy_cost # consu

        self.default

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
class EnergyProducingOrgan:

    def __init__(self, name):
        self.name = name

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
class EnergyConvertingOrgan(Organ): pass
class EnergyBallastOrgan(Organ): pass
    """
    storage_bases: list<vector>
    leakage_rates: list<float>
    
    This liver is a 
    """

class DigestiveOrgan(EnergyProducingOrgan):
    pass

class Bone(Organ): pass
class Muscle(Organ): pass
class Joint(Organ): pass