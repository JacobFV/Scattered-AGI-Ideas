from ..node import Node


class Organ(Node):

    def __init__(self, **kwargs):
        super(Organ, self).__init__(**kwargs)

    def set_organism(self, organism):
        self.organism = organism

    @property
    def unity3d_primitive(self):
        return None

    @property
    def get_name(self):
        return f'{self.organism.get_name}_{self._name}'


class NodeOrgan(Organ):

    def __init__(self, **kwargs):
        super(NodeOrgan, self).__init__(**kwargs)

        self.incoming_edge_organs = dict() # dict<str: Organ>
        self.outgoing_edge_organs = dict() # dict<str: Organ>
        self.parallel_nodes = dict() # dict<str: Organ>

        self.chemical_energy_node = None
        # this will be assigned by the organism during graph construction

    def update_parents(self):
        self._parents.clear()
        self._parents.update(self.incoming_edge_organs)
        self._parents.update(self.outgoing_edge_organs)
        self._parents.update(self.parallel_nodes)


class EdgeOrgan(NodeOrgan):

    def __init__(self, **kwargs):
        super(EdgeOrgan, self).__init__(**kwargs)

        self.src_energy_node = None # EnergyNode
        self.dst_energy_node = None # rel, EnergyNode
        self.parallel_edge_organs = dict()
        self.antiparallel_edge_organs = dict()

    def update_parents(self):
        self._parents.clear()
        self._parents['src'] = self.src_energy_node
        self._parents['dst'] = self.dst_energy_node
        self._parents.update(self.parallel_edge_organs)
        self._parents.update(self.antiparallel_edge_organs)