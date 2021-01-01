from ..energy_node import EnergyNode


class Organ(EnergyNode):

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

        self._energy_node = None

    def update_parents(self):
        self._parents.clear()
        self._parents.update(self.incoming_edge_organs)
        self._parents.update(self.outgoing_edge_organs)
        self._parents.update(self.parallel_nodes)

    def set_energy_node(self, node):
        self._energy_node = node


class EdgeOrgan(NodeOrgan):

    def __init__(self, **kwargs):
        super(EdgeOrgan, self).__init__(**kwargs)

        self.src_node = tuple('src', None)
        self.dst_node = tuple('dst', None)
        self.parallel_edge_organs = dict()
        self.antiparallel_edge_organs = dict()

    def update_parents(self):
        self._parents.clear()
        self._parents[self.src_node[0]] = self.src_node[1]
        self._parents[self.dst_node[0]] = self.dst_node[1]
        self._parents.update(self.parallel_edge_organs)
        self._parents.update(self.antiparallel_edge_organs)