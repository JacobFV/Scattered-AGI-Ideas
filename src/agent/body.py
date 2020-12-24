class Body:

    class Node:
        def __init__(self, name):
            self.name = name
            self.organs = []
            self.incoming_edges = []
            self.outgoing_edges = []

    class Edge:
        def __init__(self, src, dst):
            """
            params:
                src (Body.Node)
                dst (Body.Node)
            """
            self.src = src
            self.dst = dst
            self.organs = []

    def __init__(self, nodes, edges):
        """
        nodes (list<str>): nodes to init
        edges (list<tuple<str,str>>): edges to init
        """
        self.nodes = {name: Body.Node(name) for name in nodes}
        self.edges = {
            (src_name,dst_name):
                Body.Edge(src=self.nodes[src_name], dst=self.nodes[dst_name])
            for src_name, dst_name in edges}
        for edge in self.edges:
            edge.src.outgoing.append(edge)
            edge.dst.incoming.append(edge)

    @property
    def get_all_organs(self):
        all_organs = []
        for node in list(self.nodes.values()):
            all_organs.extend(node.organs)
        for edge in list(self.edges.values()):
            all_organs.extend(edge.organs)
        return all_organs

    @staticmethod
    def init_from_rna(agent):
        """gene expression function for body"""
        raise NotImplementedError() # TODO
