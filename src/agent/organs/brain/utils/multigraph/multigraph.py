import tensorflow as tf
keras = tf.keras
tfkl = keras.layers
K = keras.backend

class Multigraph:

    def __init__(self, Vs={}, Es={}, As={}):
        """
        Vs: dict<str, Tensor>
        rels: dict<(str,str), Tensor>

        """
        self.Vs = Vs
        self.Es = Es
        self.As = As

    @property
    def N_v(self, name):
        return tf.shape(self.Vs[name])[-2]

    @property
    def d_v(self, name):
        return tf.shape(self.Vs[name])[-1]

    @property
    def d_e(self, src, dst):
        return tf.shape(self.Es[(src, dst)])[-1]

    def build_from_list(self, listed_data):
        listed_data_iter = iter(listed_data)

        for name, V in zip(list(self.Vs.keys()), listed_data_iter):
            self.Vs[name] = V
        for rel, E in zip(list(self.Es.keys()), listed_data_iter):
            self.Es[rel] = E
        for rel, A in zip(list(self.As.keys()), listed_data_iter):
            self.As[rel] = A

    def convert_to_list(self):
        Vs = [V for g, V in self.Vs.items()]
        Es = [V for rel, E in self.Es.items()]
        As = [V for rel, A in self.As.items()]
        return Vs + Es + As

    def connect_graphs(self, src, dst, e_emb=tf.ones((1,)), density=1.0):
        leading_dims = tf.shape(self.Vs[src])[:-2]
        N_src = tf.shape(self.Vs[src])[-2:-1]
        N_dst = tf.shape(self.Vs[dst])[-2:-1]
        self.As[(src, dst)] = tf.cast(tf.random.uniform(
            tf.concat([leading_dims, N_src, N_dst], axis=0)) < density,
                                      keras.backend.floatx())
        self.Es[(src, dst)] = tf.einsum('...sd,v->...sdv',
                                        self.As[(src, dst)], e_emb)

    def add_root_network(self,
                         root_name,
                         intragraph_density=1.0,
                         intergraph_density=1.0,
                         neighbors=[],
                         connection_direction=["src", "dst"],
                         N_v=1,
                         d_v=1):
        """convenience function to make root node network
        and connect to other graphs. Root networks provide an
        information highway for intragraph vert updates and
        can be used to connect heterogenous graphs.

        WARNING: the multigraph must have at least one other
        set of verts so we can detirmine batch size and time
        steps (or any other leading dimensions).

        neighbors (list<str>): neighboring graphs (if any) to
            connect new root network to. "src" means the root
            is a source and the neighbors are destination graphs.
            "dst" means the root graph is an innode in the graph
            network. Both directions can be specified.
        """

        # create root graph verts
        leading_dims = tf.shape(list(self.Vs.values())[0])[:-2]
        self.Vs[root_name] = tf.zeros(tf.concat([leading_dims,
                                                 tf.TensorShape((N_v, d_v))], axis=0))

        # connect graph internally
        self.connect_graphs(root_name, root_name,
                            density=intragraph_density)

        # connect with neighbors
        for neighbor in neighbors:
            if "src" in connection_direction:
                self.connect_graphs(root_name, neighbor,
                                    density=intergraph_density)
            if "dst" in connection_direction:
                self.connect_graphs(neighbor, root_name,
                                    density=intergraph_density)

    def batch_size_multigraph(self, batch_size=1):
        """
        This function can only be called once. It transforms a normal multigraph
        into one with multiple batches. This function is useful for varying batch sizes
        perhaps training and inference with the multigraph_update_cell.
        """

        # NOTE keras only exposes axes[1:] during `call` so this code cannot help
        # no ACTUALLY keras performs elementwise operations on the call so just match
        # your elements with the keras symbolic tensors
        # raise Exception('dont use this')
        for g, V in self.Vs.items():
            self.Vs[g] = tf.tile(tf.expand_dims(V, axis=0), tf.constant([batch_size, 1, 1]))
        for rel, E in self.Es.items():
            self.Es[rel] = tf.tile(tf.expand_dims(E, axis=0), tf.constant([batch_size, 1, 1, 1]))
        for rel, A in self.As.items():
            self.As[rel] = tf.tile(tf.expand_dims(A, axis=0), tf.constant([batch_size, 1, 1]))

    def make_keras_compat(self):
        for g, V in self.Vs.items():
            self.Vs[g] = K.variable(V)
        for rel, E in self.Es.items():
            self.Es[rel] = K.variable(E)
        for rel, A in self.As.items():
            self.As[rel] = K.variable(A)