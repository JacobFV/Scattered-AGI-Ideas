import itertools

import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
keras = tf.keras
tfkl = keras.layers

from graph_layer import GraphLayer
from multigraph import Multigraph
from multigraph_layer import MultigraphLayer


def dense_to_dense_multigraph_net(d_in, d_out, B, N_v=64, d_v=16, d_e=8):
    """convenience initializer for MHA graph RNN with dense input and output
    This layer actually uses python code to process time steps, but it sandwhiches
    that between input and output keras tensors
    """

    multigraph = Multigraph(
        Vs={"cell": tf.zeros((N_v, d_v))},
        Es={("cell", "cell"): tf.zeros((N_v, N_v, d_e))},
        As={("cell", "cell"): tf.zeros((N_v, N_v))})
    multigraph.add_root_network(
        root_name="in",
        intragraph_density=0.0,
        intergraph_density=1.0,
        neighbors=["cell"],
        connection_direction=["src"],
        N_v=1,
        d_v=d_in)
    multigraph.add_root_network(
        root_name="out",
        intragraph_density=0.0,
        intergraph_density=1.0,
        neighbors=["cell"],
        connection_direction=["dst"],
        N_v=1,
        d_v=d_out)
    multigraph.batch_size_multigraph(B)
    # multigraph.make_keras_compat()
    multigraph_net = MultigraphLayer(
        multigraph_template=multigraph,
        f_rel_update={
            # the input layer
            ("in", "cell"): GraphLayer(
                name="in2cell",
                f_inp=GraphLayer.f_inp_concat(),
                f_pool=GraphLayer.f_pool_attn(
                    d_key=8, d_val=16, N_heads=8),
                f_v_up=GraphLayer.f_v_up_beta(),
                f_e_up=GraphLayer.f_e_up_dense(),
                f_adj_up=GraphLayer._f_adj_up()),
            # the working memory layer
            ("cell", "cell"): GraphLayer(
                name="cell2cell",
                f_inp=GraphLayer.f_inp_concat(),
                f_pool=GraphLayer.f_pool_attn(
                    d_key=16, d_val=64, N_heads=8),
                f_v_up=GraphLayer.f_v_up_beta(),
                f_e_up=GraphLayer.f_e_up_attn(
                    d_key=8, d_val=16, N_heads=8),
                f_adj_up=GraphLayer._f_adj_up()),
            # the output layer
            ("cell", "out"): GraphLayer(
                name="cell2out",
                f_inp=GraphLayer.f_inp_concat(),
                f_pool=GraphLayer.f_pool_attn(
                    d_key=8, d_val=16, N_heads=8),
                f_v_up=GraphLayer.f_v_up_beta(),
                f_e_up=GraphLayer.f_e_up_attn(
                    d_key=4, d_val=8, N_heads=8),
                f_adj_up=GraphLayer._f_adj_up())},
        f_inp=MultigraphLayer.f_inp_update_root("in"),
        f_update_seq=(lambda x: [
            ("in", "cell"),
            ("cell", "cell"),
            ("cell", "out")]),
        f_ret=MultigraphLayer.f_ret_just_root("out"))

    return multigraph_net


def dense_to_dense_recurrent_multigraph_net(d_in, d_out, B, T, N_v=64, d_v=16, d_e=8):
    multigraph_net = dense_to_dense_multigraph_net(d_in, d_out, B, N_v, d_v, d_e)
    recurrent_multigraph_net = tfkl.RNN(multigraph_net)
    return recurrent_multigraph_net


d_in = 24
d_out = 32
B = 4
T0 = 20


def get_data(B, T, d_in, d_out):
    X = 1.4 + 2 * tf.random.normal((B, T, d_in))
    Y_true = X @ tf.random.uniform((d_in, d_out))
    return X, Y_true


dataset = [
    get_data(B=B, T=round((batch_num ** 0.25 + 1) * T0), d_in=d_in, d_out=d_out)
    for batch_num in range(20)]

rmgnet = dense_to_dense_recurrent_multigraph_net(
    d_in, d_out, B, T, N_v=64, d_v=16, d_e=8)

y_pred = rmgnet(dataset[0])