class Body:

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

    def get_incoming_edges(self, node): pass

    def get_outgoing_edges(self, node): pass

    def get_organs_at_node(self, node): pass

    def get_organs_at_edge(self, src_node, dst_node): pass

    def get_node_for_organ(self, organ): pass

    def get_edge_for_organ(self, organ): pass  # returns tuple<str,str>

    def get_all_organs(self): pass