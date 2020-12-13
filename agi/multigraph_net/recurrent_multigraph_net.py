import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
keras = tf.keras
tfkl = keras.layers

from graph_net import GraphNet
from multigraph import Multigraph
from multigraph_net import MultigraphNet

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
    multigraph_net = MultigraphNet(
        multigraph_template=multigraph,
        f_rel_update={
            # the input layer
            ("in", "cell"): GraphNet(
                name="in2cell",
                f_inp=GraphNet.f_inp_concat(),
                f_pool=GraphNet.f_pool_attn(
                    d_key=8, d_val=16, N_heads=8),
                f_v_up=GraphNet.f_v_up_beta(),
                f_e_up=GraphNet.f_e_up_dense(),
                f_adj_up=GraphNet._f_adj_up()),
            # the working memory layer
            ("cell", "cell"): GraphNet(
                name="cell2cell",
                f_inp=GraphNet.f_inp_concat(),
                f_pool=GraphNet.f_pool_attn(
                    d_key=16, d_val=64, N_heads=8),
                f_v_up=GraphNet.f_v_up_beta(),
                f_e_up=GraphNet.f_e_up_attn(
                    d_key=8, d_val=16, N_heads=8),
                f_adj_up=GraphNet._f_adj_up()),
            # the output layer
            ("cell", "out"): GraphNet(
                name="cell2out",
                f_inp=GraphNet.f_inp_concat(),
                f_pool=GraphNet.f_pool_attn(
                    d_key=8, d_val=16, N_heads=8),
                f_v_up=GraphNet.f_v_up_beta(),
                f_e_up=GraphNet.f_e_up_attn(
                    d_key=4, d_val=8, N_heads=8),
                f_adj_up=GraphNet._f_adj_up())},
        f_inp=MultigraphNet.f_inp_update_root("in"),
        f_update_seq=(lambda x: [
            ("in", "cell"),
            ("cell", "cell"),
            ("cell", "out")]),
        f_ret=MultigraphNet.f_ret_just_root("out"))

    return multigraph_net

def make_dense_to_dense_multigraph_net_recurrent(multigraph_net, T, d_in):
    """
    cell_model: x, multigraph -> y, multigraph'
    """
    input_layer = tfkl.Input((T,d_in))
    inputs = tfkl.Lambda(lambda x: tf.unstack(value=x, axis=1))(input_layer)
    multigraph = multigraph_net.multigraph_template
    outputs = []
    for t in tf.range(T):
        output, multigraph = multigraph_net(inputs[t], multigraph)
        outputs.append(output)

    output_layer = tfkl.concatenate(outputs, axis=1)
    return keras.Model(inputs=input_layer, outputs=output_layer)

def dense_to_dense_recurrent_multigraph_net(d_in, d_out, B, T, N_v=64, d_v=16, d_e=8):
    multigraph_net = dense_to_dense_multigraph_net(d_in, d_out, B, N_v, d_v, d_e)
    recurrent_multigraph_net = make_dense_to_dense_multigraph_net_recurrent(multigraph_net, T, d_in)
    return recurrent_multigraph_net


d_in=24
d_out=32
B = 4
T = 20

X = 1.4 + 2*tf.random.normal((B,T,d_in))
Y_true = tf.reduce_sum(X, axis=1)

mgnet = dense_to_dense_multigraph_net(
    d_in=d_in, d_out=d_out, B=B)

y_pred = []
multigraph = mgnet.multigraph_template
for t in range(T):
    output, multigraph = mgnet(X[:,t,:], multigraph)
    y_pred.append(output)

Y_pred = tf.stack(y_pred, axis=1)
print(Y_pred)
tf.print(Y_pred[0])

"""
recurrent_multigraph_net = dense_to_dense_recurrent_multigraph_net(
    d_in=d_in, d_out=d_out, B=B, T=T)
Y_pred = recurrent_multigraph_net(X)
print(Y_pred)"""

"""model = keras.Sequential()
model.add(tfkl.Input((T,d_in)))
model.add(rnn)
model.add(tfkl.Dense(d_out))
model.compile('Adam', 'mse', ['accuracy'])

Y_pred = model.predict(X)
model.evaluate(X, Y_true)
model.fit(X, Y_true)
model.evaluate(X, Y_true)
"""