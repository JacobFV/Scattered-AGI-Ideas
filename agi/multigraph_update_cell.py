from multigraph import Multigraph

import random

import tensorflow as tf
keras = tf.keras
tfkl = keras.layers


class MultigraphUpdateCell(tfkl.Layer):

    def __init__(self,
                 multigraph_template: Multigraph,
                 f_rel_update,
                 f_inp=(lambda inp, mg: mg),
                 f_update_seq=None,
                 f_ret=(lambda x: x),
                 randomized_update_seq=False,
                 **kwargs):
        """
        multigraph_template (Multigraph): multigraph with same structure as the multigraph
            used during inference. The batch dimension (0) of the tensors here should exactly
            match the batch size used for training or inference
        f_rel_update (dict<(str,str): GraphNet>): update functions
            for each source-destination graph pairs
        f_inp (input, multigraph -> multigraph): multigraph input modifications
        f_update_seq (multigraph -> list<(str, str)>): detirmines the order
            multigraph (src, dst) pair updates should take place
        f_ret (multigraph -> obj): return function from multigraph update call
        randomized_update_seq (bool): whether to randomize the order of (src,dst)
            relation updates.
        """
        super(MultigraphUpdateCell, self).__init__(**kwargs)
        self.f_rel_update = f_rel_update
        self.f_inp = f_inp
        if f_update_seq is None:
            f_update_seq = MultigraphUpdateCell.f_update_seq_egocentric
        self.f_update_seq = f_update_seq
        self.f_ret = f_ret
        self.randomized_update_seq = randomized_update_seq

        self.multigraph_state = multigraph_template
        self.state_size = (1,)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return tf.zeros(self.state_size)

    def call(self, input_at_t, states_at_t, training=False):
        multigraph = self.multigraph_state

        multigraph = self.f_inp(input_at_t, multigraph)
        for rel in self.f_update_seq(multigraph):
            src, dst = rel
            multigraph.V[dst], multigraph.E[rel], multigraph.A[rel] = \
                self.f_rel_update[rel]([
                    multigraph.Vs[src], multigraph.Vs[dst],
                    multigraph.Es[rel], multigraph.As[rel]])

        self.multigraph_state = multigraph
        return self.f_ret(multigraph), states_at_t

    @staticmethod
    class f_inp_update_root:
        def __init__(self, root_name):
            self.root_name = root_name

        def __call__(self, inputs, multigraph):
            #multigraph.Vs[self.root_name][..., 0, :] = inputs
            multigraph.Vs[self.root_name] = tf.reshape(inputs,
                shape=(multigraph.Vs[self.root_name].shape[0], 1, -1))
            return multigraph

    @staticmethod
    def f_update_seq_reg(multigraph):
        """just go through all defined relations"""
        seq = list(multigraph.Vs.keys())
        if multigraph.randomized_update_seq:
            random.shuffle(seq)
        return seq

    @staticmethod
    def f_update_seq_egocentric(multigraph):
        """first perform intragraph update, then intergraph update"""

        all_pairs = list(multigraph.Es.keys())
        if multigraph.randomized_update_seq:
            random.shuffle(all_pairs)

        intragraph_pairs = [(src, dst) for (src, dst)
                            in all_pairs if src == dst]
        intergraph_pairs = [(src, dst) for (src, dst)
                            in all_pairs if src != dst]

        return intragraph_pairs + intergraph_pairs

    @staticmethod
    class f_ret_just_graph:
        def __init__(self, graph_name):
            self.graph_name = graph_name

        def __call__(self, multigraph):
            return (multigraph.Vs[self.graph_name],
                    multigraph.Es[(self.graph_name, self.graph_name)],
                    multigraph.As[(self.graph_name, self.graph_name)])

    @staticmethod
    class f_ret_just_root:
        def __init__(self, root_name):
            self.root_name = root_name

        def __call__(self, multigraph):
            return tf.reduce_mean(
                multigraph.Vs[self.root_name],
                axis=-2)