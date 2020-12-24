from gene_expression import gene_loader


class Agent:

    class Morphology:

        def __init__(self):
            self.nodes = dict()
            self.edges = dict()

        def add_node(self, name): pass
        def add_edge(self, src_node, dst_node): pass
        def add_organ_to_node(self, name): pass
        def add_organ_to_edge(self, src_node, dst_node): pass

        def remove_node(self, name): pass
        def remove_edge(self, src_node, dst_node): pass
        def remove_organ_from_node(self, name): pass
        def remove_organ_from_edge(self, src_node, dst_node): pass

        def incoming_edges(self, node): pass
        def outgoing_edges(self, node): pass

        def organs_at_node(self, node): pass
        def organs_at_edge(self, src_node, dst_node): pass
        def node_for_organ(self, organ): pass
        def edge_for_organ(self, organ): pass # returns tuple<str,str>


    def __init__(self,
                 name,
                 dna,
                 simulator_ip=None,
                 orchestrator_ip=None):

        self._name = name
        self.simulator_ip = simulator_ip
        self.orchestrator_ip = orchestrator_ip

        self.dna = dna
        self.rna = None
        self.gene_expression_fns = [ gene_loader ]

        self.morphology = Agent.Morphology()

    def step(self):
        pass

    def train(self):
        pass

    @property
    def get_name(self):
        return self._name

    def add_to_simulation_env(self):
        raise NotImplementedError()

    def remove_from_simulation_env(self):
        raise NotImplementedError()