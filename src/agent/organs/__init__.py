from .. import utils
from .brain.nerves import Sensor, Actuator


class Organ(utils.Freezable,
            utils.Stepable,
            utils.Trainable,
            utils.PermanentName,
            Sensor,
            Actuator):

    def __init__(self, agent, **kwargs):
        super(Organ, self).__init__(**kwargs)
        self.agent = agent

    @property
    def unity3d_primitive(self):
        raise NotImplementedError()

class NodeOrgan(Organ):

    def __init__(self, node, **kwargs):
        super(NodeOrgan, self).__init__(**kwargs)
        self.nodes = node

class EdgeOrgan(Organ):

    def __init__(self, edge, **kwargs):
        super(EdgeOrgan, self).__init__(**kwargs)
        self.edge = edge