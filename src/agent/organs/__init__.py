from .. import utils
from .brain.nerves import Sensor, Actuator
from ..gene_expression import SimpleTranscription, GeneLoader


class Organ(utils.Freezable,
            utils.Stepable,
            utils.Trainable,
            utils.SimulationEnvCommunicator,
            utils.PermanentName,
            Sensor,
            Actuator):

    def step(self):
        """first perform gene expression, then step through any substepables"""
        for gene_expression_fn in self.gene_expression_fns[:]:
            gene_expression_fn.step()
        super(Organ, self).step()

    def __init__(self, agent, dna, **kwargs):

        super(Organ, self).__init__(**kwargs)

        self.agent = agent
        self.dna = dna
        self.gene_expression_fns = [
            SimpleTranscription(agent=self.agent),
            GeneLoader(agent=self.agent)
        ]

    @property
    def unity3d_primitive(self):
        raise NotImplementedError()

class NodeOrgan(Organ):

    def __init__(self, **kwargs):
        super(NodeOrgan, self).__init__(**kwargs)
        self.incoming_edges = []
        self.outgoing_edges = []


class EdgeOrgan(Organ):

    def __init__(self, **kwargs):
        super(EdgeOrgan, self).__init__(**kwargs)
        self.src_nodes = []
        self.dst_nodes = []