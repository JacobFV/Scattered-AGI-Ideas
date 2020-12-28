from .. import utils
from .brain.nerves import Sensor, Actuator


class Organ(utils.Freezable,
            utils.Stepable,
            utils.Trainable,
            utils.SimulationEnvCommunicator,
            utils.PermanentName,
            Sensor,
            Actuator):

    def _set_agent(self, organism):
        self.organism = organism

    @property
    def unity3d_primitive(self):
        raise NotImplementedError()

    @property
    def get_name(self):
        return f'{self.organism.get_name}_{self._name}'

class NodeOrgan(Organ):

    def __init__(self, **kwargs):
        super(NodeOrgan, self).__init__(**kwargs)

    def _set_node(self, node):
        self.node = node


class EdgeOrgan(Organ):

    def __init__(self, **kwargs):
        super(EdgeOrgan, self).__init__(**kwargs)

    def _set_nodes(self, src, dst):
        self.src = src
        self.dst = dst