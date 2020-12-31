from .. import utils
from .brain.nerves import Sensor, Actuator


class Organ(utils.Freezable,
            utils.Stepable,
            utils.SimulationEnvCommunicator,
            utils.PermanentName,
            Sensor,
            Actuator):

    def set_organism(self, organism):
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

        self.node = None

        # convenience variables for subclasses
        self.parallel_node_organs = []
        self.incoming_edge_organs = []
        self.outgoing_edge_organs = []

    def set_node(self, node):
        self.node = node


class EdgeOrgan(Organ):

    def __init__(self, **kwargs):
        super(EdgeOrgan, self).__init__(**kwargs)

        self.src = None
        self.dst = None

        # convenience variables for subclasses
        self.parallel_edges = []
        self.antiparallel_edges = []

    def set_nodes(self, src, dst):
        self.src = src
        self.dst = dst