import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers

#import tensorflow_probability as tfp
#tfd = tfp.distributions
#tfpl = tfp.layers
#tfb = tfp.bijectors

import random

class GraphNet(tfk.Model):
    
    def __init__(self,
        f_inp,
        f_pool,
        f_v_up,
        f_e_up,
        f_adj_up,
        **kwargs):
        """
        """
        
        super(GraphNet, self).__init__(kwargs)
        
        self.f_inp = f_inp
        self.f_pool = f_pool
        self.f_v_up = f_v_up
        self.f_e_up = f_e_up
        self.f_adj_up = f_adj_up
        
        self.f_V_src_loc = tfkl.Lambda(lambda V_src, A: 
            tf.einsum('...sv,...sd->...sdv', V_src, A))
        self.f_V_dst_loc = tfkl.Lambda(lambda V_dst, A: 
            tf.einsum('...dv,...sd->...dsv', V_dst, A))
        self.f_perm = tfkl.Lambda(lambda x: tf.einsum('...sdv->...dsv', x))
        
    def call(self, inputs, training=False):
        """
        """
        
        # unpack inputs
        V_src, V_dst, E, A = inputs
        
        # provide vert-localized copies of src and dst verts
        V_src_loc = self.f_V_src_loc(V_src, A)
        V_dst_loc = self.f_V_dst_loc(V_dst, A)
        
        # get src-dst pair-specific inputs to dst verts
        inp = self.f_inp([V_src_loc, E], training=training)
        inp = self.f_perm(inp)
        
        # pool src-dst pair-specific inputs
        V_dst_new = self.f_pool([V_dst, inp], training=training)    
        
        # update dst verts
        V_dst = self.f_v_up([V_dst, V_dst_new], training=training)
        
        # update edges
        E = self.f_e_up([V_src_loc, V_dst_loc, E], training=training)
        
        # update adjacency matrix
        A = self.f_adj_up(E, training=training)
        
        return V_dst, E, A
    
        
    @staticmethod
    def f_pool_sum():
        return tfkl.Lambda(lambda V_dst, inp: tf.reduce_sum(inp, axis=-2))
    @staticmethod
    def f_pool_ave():
        return tfkl.Lambda(lambda V_dst, inp: tf.reduce_mean(inp, axis=-2))
    @staticmethod
    def f_pool_prod():
        return tfkl.Lambda(lambda V_dst, inp: tf.reduce_prod(inp, axis=-2))

    class f_pool_attn(tfkl.Layer):
        
        def __init__(self, d_key=8, d_val=None, N_heads=8, pre_layer_normalization=True, **kwargs):
            """
            pre-LN (https://arxiv.org/abs/2004.08249)
            """
            super(GraphNet.f_pool_attn, self).__init__(**kwargs)
            
            self.pre_layer_normalization = pre_layer_normalization                
            self.d_key = d_key
            self.d_val = self.d_key if d_val is None else d_val
            self.N_heads = N_heads
            
            if self.pre_layer_normalization:
                self.V_dst_LN = tfkl.LayerNormalization()
                self.inp_LN = tfkl.LayerNormalization()
            
        def build(self, input_shape):
            V_dst_shape, inp_shape = input_shape
            
            self.f_val = tfkl.Dense(self.N_heads * self.d_val, 'relu')
            self.f_key = tfkl.Dense(self.N_heads * self.d_key, 'relu')
            self.f_query = tfkl.Dense(self.N_heads * self.d_key, 'relu')
            
            self.reshape_q = tfkl.Reshape(V_dst_shape[:-2] +
                (self.N_heads, self.d_key))
            self.reshape_k = tfkl.Reshape(inp_shape[:-2] +
                (self.N_heads, self.d_key))
            self.reshape_v = tfkl.Reshape(inp_shape[:-2] +
                (self.N_heads, self.d_val))
            
            def _f_MHA(queries, keys, values):
                score = tf.einsum('...dhq,...dshq->dsh', queries, keys)
                score = score / tf.sqrt(self.d_key)
                score = tf.nn.softmax(score, axis=-1)
                return tf.einsum('...dsh,...dshv->...dhv', score, values)
            self.f_MHA = tfkl.Lambda(lambda q,k,v: _f_MHA(q,k,v))
            
            self.f_cat = tfkl.Reshape(V_dst_shape[:-1]+(-1,))
            self.f_emb_cat = tfkl.Dense(V_dst_shape[-1], 'relu')
        
        def call(self, inputs, training=False):
            # unpack inputs
            V_dst, inp = inputs

            # pre-LN
            if self.pre_layer_normalization:
                V_dst = self.V_dst_LN(V_dst, training=training)
                inp = self.inp_LN(inp, training=training)
            
            # generate queries, keys, and values for all heads
            queries = self.f_query(V_dst, training=training)  # [..., N_dst, N_heads*d_key]
            keys = self.f_key(inp, training=training) # [..., N_dst, N_src, N_heads*d_key]
            values = self.f_val(inp, training=training) # [..., N_dst, N_src, N_heads*d_val]
            
            # reshape into separate heads
            queries = self.reshape_q(queries) # [..., N_dst, N_heads, d_key]
            keys = self.reshape_k(keys) # [..., N_dst, N_heads, d_key]
            values = self.reshape_v(values) # [..., N_dst, N_heads, d_key]
            
            # perform multi-head attention
            mha_lookup = self.f_MHA([queries, keys, values], training=training)
            # [..., N_dst, N_heads, d_val]
            
            # concatenate heads
            mha_cat = self.f_cat(mha_lookup, training=training)
            
            # embed in output space
            return self.f_emb_cat(mha_cat, training=training)

    @staticmethod
    def f_v_up_add():
        return tfkl.Add()

    @staticmethod
    def f_v_up_direct():
        return tfkl.Lambda(lambda V_dst, V_dst_new: V_dst_new)

    class f_v_up_beta(tfkl.Layer):
        def __init__(self, **kwargs):
            super(GraphNet.f_v_up_beta, self).__init__(**kwargs)
            self.f_beta = tfkl.Dense(1, 'softmax')
        def call(self, inputs, training=False):
            V_dst, V_dst_new = inputs
            beta = self.f_beta(V_dst_new)
            return beta*V_dst + (1-beta)*V_dst_new
        
    class f_v_up_alphabeta(tfkl.Layer):
        def __init__(self, **kwargs):
            super(GraphNet.f_v_up_alphabeta, self).__init__(**kwargs)
            self.f_beta = tfkl.Dense(1, 'softmax')
            self.f_alpha = tfkl.Dense(1, 'softmax')
        def call(self, inputs, training=False):
            V_dst, V_dst_new = inputs
            alpha = self.f_alpha(V_dst)
            beta = self.f_beta(V_dst_new)
            return alpha*V_dst + beta*V_dst_new

    @staticmethod
    def f_inp_concat():
        return tfkl.Concatenate()

    @staticmethod
    def f_inp_edges():
        return tfkl.Lambda(lambda V_src_loc, E: E)

    @staticmethod
    def f_inp_verts():
        return tfkl.Lambda(lambda V_src_loc, E: V_src_loc)

    @staticmethod
    def _f_adj_up():
        def f(x):
            y=tfkl.Dense(1, 'softmax')(x)
            y=tf.squeeze(y)
            
        return tfkl.Lambda(lambda E: f(E))

    @staticmethod
    def f_e_up_const():
        return tfkl.Lambda(lambda V_src_loc, V_dst_loc, E: E)

    class f_e_up_dense(tfkl.Layer):
        def __init__(self, **kwargs):
            super(GraphNet.f_e_up_dense, self).__init__(**kwargs)
        def build(self, input_shape):
            V_src_loc_shape, V_dst_loc_shape, E_shape = input_shape
            self.f_E_new = tfkl.Dense(tf.shape(E_shape)[-1], 'relu')
            self.V_dst_perm = tfkl.Lambda(
                lambda x: tf.einsum('...dsv->...sdv', x))
        def call(self, inputs, training=False):
            V_src_loc, V_dst_loc, E = inputs
            V_dst_loc_perm = self.V_dst_perm(V_dst_loc)
            return self.f_E_new(tfkl.concatenate([
                V_src_loc, V_dst_loc_perm, E]))
        
    class f_e_up_dense_oneway(tfkl.Layer):
        def __init__(self, **kwargs):
            super(GraphNet.f_e_up_dense_oneway, self).__init__(**kwargs)
        def build(self, input_shape):
            V_src_loc_shape, V_dst_loc_shape, E_shape = input_shape
            self.V_dst_perm = tfkl.Lambda(
                lambda x: tf.einsum('...dsv->...sdv', x))
            self.f_E_new = tfkl.Dense(tf.shape(E_shape)[-1], 'relu')
        def call(self, inputs, training=False):
            V_src_loc, V_dst_loc, E = inputs
            return self.f_E_new(tfkl.concatenate([V_src_loc, E]))
        
    class f_e_up_beta(tfkl.Layer):
        def __init__(self, **kwargs):
            super(GraphNet.f_e_up_beta, self).__init__(**kwargs)
        def build(self, input_shape):
            V_src_loc_shape, V_dst_loc_shape, E_shape = input_shape
            self.V_dst_perm = tfkl.Lambda(
                lambda x: tf.einsum('...dsv->...sdv', x))
            self.f_beta = tfkl.Dense(1, 'softmax')
            self.f_E_new = tfkl.Dense(tf.shape(E_shape)[-1], 'relu')
        def call(self, inputs, training=False):
            V_src_loc, V_dst_loc, E = inputs
            V_dst_loc_perm = self.V_dst_perm(V_dst_loc)
            E_new = self.f_E_new(tfkl.concatenate([
                V_src_loc, V_dst_loc_perm, E]))
            beta = self.f_beta(tfkl.concatenate([V_src_loc, V_dst_loc_perm]))
            return beta*V_dst_loc_perm + (1-beta)*E_new

    class f_e_up_attn(tfkl.Layer):
        
        def __init__(self, d_key=8, d_val=None, N_heads=8, pre_layer_normalization=True, **kwargs):
            """
            pre-LN (https://arxiv.org/abs/2004.08249)
            """
            super(GraphNet.f_e_up_attn, self).__init__(**kwargs)
            
            self.pre_layer_normalization = pre_layer_normalization                
            self.d_key = d_key
            self.d_val = self.d_key if d_val is None else d_val
            self.N_heads = N_heads
            
            if self.pre_layer_normalization:
                self.V_dst_LN = tfkl.LayerNormalization()
                self.inp_LN = tfkl.LayerNormalization()
            
        def build(self, input_shape):
            V_src_loc_shape, V_dst_loc_shape, E_shape = input_shape
            
            self.V_dst_perm = tfkl.Lambda(
                lambda x: tf.einsum('...dsv->...sdv', x))
            
            self.cat_q_data = tfkl.Concatenate()
            self.cat_kv_data = tfkl.Concatenate()
            
            self.f_val = tfkl.Dense(self.N_heads * self.d_val, 'relu')
            self.f_key = tfkl.Dense(self.N_heads * self.d_key, 'relu')
            self.f_query = tfkl.Dense(self.N_heads * self.d_key, 'relu')
            
            self.reshape_q = tfkl.Reshape(E_shape[:-1] +
                (self.N_heads, self.d_key))
            self.reshape_k = tfkl.Reshape(E_shape[:-1] +
                (self.N_heads, self.d_key))
            self.reshape_v = tfkl.Reshape(E_shape[:-1] +
                (self.N_heads, self.d_val))
            
            def _f_MHA(queries, keys, values):
                score = tf.einsum('...sdhq,...sdhq->sdh', queries, keys)
                score = score / tf.sqrt(self.d_key)
                score = tf.nn.softmax(score, axis=-1)
                return tf.einsum('...sdh,...sdhv->...dhv', score, values)
            self.f_MHA = tfkl.Lambda(lambda q,k,v: _f_MHA(q,k,v))
            
            self.f_cat = tfkl.Reshape(E_shape[:-1]+(-1,))
            self.f_emb_cat = tfkl.Dense(E_shape[-1], 'relu')
        
        def call(self, inputs, training=False):
            # unpack inputs
            V_src_loc, V_dst_loc, E = inputs

            # pre-LN
            if self.pre_layer_normalization:
                V_dst = self.V_dst_LN(V_dst, training=training)
                inp = self.inp_LN(inp, training=training)
            
            V_dst_loc_perm = self.V_dst_perm(V_dst_loc)
            
            q_data = self.cat_q_data([V_dst_loc, E])
            kv_data = self.cat_kv_data([V_src_loc, E])
            
            # generate queries, keys, and values for all heads
            queries = self.f_query(q_data, training=training)  # [..., N_src, N_dst, N_heads*d_key]
            keys = self.f_key(kv_data, training=training) # [..., N_src, N_dst, N_heads*d_key]
            values = self.f_val(kv_data, training=training) # [..., N_src, N_dst, N_heads*d_val]
            
            # reshape into separate heads
            queries = self.reshape_q(queries) # [..., N_src, N_dst, N_heads, d_key]
            keys = self.reshape_k(keys) # [..., N_src, N_dst, N_heads, d_key]
            values = self.reshape_v(values) # [..., N_src, N_dst, N_heads, d_key]
            
            # perform multi-head attention
            mha_lookup = self.f_MHA([queries, keys, values], training=training)
            # [..., N_src, N_dst, N_heads, d_val]
            
            # concatenate heads
            mha_cat = self.f_cat(mha_lookup, training=training)
            # [..., N_src, N_dst, N_heads*d_val]
            
            # embed in output space
            return self.f_emb_cat(mha_cat, training=training)
            # [..., N_src, N_dst, d_E]

class MultiGraph:
    
    def __init__(self, Vs=dict(), Es=dict(), As=dict()):
        """
        Vs: dict<str, Tensor>
        rels: dict<(str,str), Tensor>
        
        """
        self.Vs = Vs
        self.Es = Es
        self.As = As
    
    def to_dict(self):
        return {
            "Vs": self.Vs,
            "Es": self.Es,
            "As": self.As
        }
    
    @staticmethod
    def from_dict(dict):
        return MultiGraph(
            Vs=dict["Vs"],
            Es=dict["Es"],
            As=dict["As"])

    def to_list(self):
        return [
            self.Vs[k] for k in list(self.Vs.keys())
        ] + [
            self.Es[k] for k in list(self.Es.keys())
        ] + [
            self.As[k] for k in list(self.As.keys())
        ]
    
    def load_from_list(self, list_inp):

        i = 0

        for k in list(self.Vs.keys()):
            self.Vs[k] = list_inp[i]
            i = i + 1

        for k in list(self.Es.keys()):
            self.Es[k] = list_inp[i]
            i = i + 1

        for k in list(self.As.keys()):
            self.As[k] = list_inp[i]
            i = i + 1
    
    @property
    def N_v(self, name):
        return tf.shape(self.Vs[name])[-2]
    @property
    def d_v(self, name):
        return tf.shape(self.Vs[name])[-1]
    @property
    def d_e(self, src, dst):
        return tf.shape(self.Es[(src, dst)])[-1]
    
    def connect_graphs(self, src, dst, e_emb=tf.ones((1,)), density=1.0):
        leading_dims = tf.shape(self.Vs[src])[:-2]
        N_src = tf.shape(self.Vs[src])[-2:-1]
        N_dst = tf.shape(self.Vs[dst])[-2:-1]
        self.As[(src,dst)] = tf.cast(tf.random.uniform(
            tf.concat([leading_dims, N_src, N_dst], axis=0)) < density,
            tfk.backend.floatx())
        self.Es[(src,dst)] = tf.einsum('...sd,v->...sdv',
            self.As[(src,dst)], e_emb)
    
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
        print("new shape: ", tf.concat([leading_dims,
                tf.TensorShape((N_v, d_v))], axis=0))
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

class MultiGraphRNNCell(tfkl.Layer):
    
    def __init__(self,
        multigraph_template,
        f_rel_update,
        f_inp=(lambda inp, mg: mg),
        f_update_seq=None,
        f_ret=(lambda x: x),
        randomized_update_seq=False):
        """
        f_rel_update (dict<(str,str): GraphNet): update functions
            for each source-destination graph pairs. If `None`, specify
            an `f_rel_update_model` that will be applied to all edges in
            the multigraph.
        f_rel_update_model (GraphNet): updating function to be copied
            for all source-destination graph relations in the case that
            `f_rel_update` is `None`.
        """
        self.multigraph_template = multigraph_template
        self.f_rel_update = f_rel_update
        self.f_inp = f_inp
        if f_update_seq is None:
            f_update_seq = MultiGraphRNNCell.f_update_seq_egocentric
        self.f_update_seq = f_update_seq
        self.f_ret = f_ret
        
        ## RNNCell attributes
        self.state_size = tf.shape(multigraph_template.to_list())
        
        f_ret_template = f_ret(multigraph_template)
        if isinstance(f_ret_template, MultiGraph):
            f_ret_template = f_ret_template.to_dict()
        self.output_size = tf.shape(f_ret_template)

        self.randomized_update_seq = randomized_update_seq
    
    def call(self, input_at_t, state_at_t, training=False):
        multigraph = self.multigraph_template.load_from_list(state_at_t)
        multigraph = self.f_inp(input_at_t, multigraph)
        for rel in self.f_update_seq(multigraph):
            src, dst = rel
            multigraph.V[dst], multigraph.E[rel], multigraph.A[rel] = \
                self.f_rel_update[rel](
                    multigraph.V[src], multigraph.V[dst],
                    multigraph.E[rel], multigraph.A[rel])
        return self.f_ret(multigraph), multigraph.to_list()
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.multigraph_template.to_list()
    
    @staticmethod
    class f_inp_update_root:
        def __init__(self, root_name):
            self.root_name = root_name
        def __call__(self, inputs, multigraph):
            multigraph.Vs[self.root_name][...,0,:]=inputs
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
            
        intragraph_pairs = [(src,dst) for (src,dst)
                            in all_pairs if src==dst]
        intergraph_pairs = [(src,dst) for (src,dst)
                            in all_pairs if src!=dst]
                    
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

def dense2denseRNN(d_in, d_out, N_v=64, d_v=16, d_e=8):
    """Convenience initializer for MHA graph RNN with dense input and output
    """
    
    mg = MultiGraph(
        Vs={"cell":tf.zeros((N_v, d_v))},
        Es={"cell":tf.zeros((N_v, N_v, d_v))},
        As={"cell":tf.zeros((N_v, N_v))})
    mg.add_root_network(
        root_name="in",
        intragraph_density=0.0,
        intergraph_density=1.0,
        neighbors=["cell"],
        connection_direction=["src"],
        N_v=1,
        d_v=d_in)
    mg.add_root_network(
        root_name="out",
        intragraph_density=0.0,
        intergraph_density=1.0,
        neighbors=["cell"],
        connection_direction=["dst"],
        N_v=1,
        d_v=d_out)
    
    rnn = tfkl.RNN(MultiGraphRNNCell(
        multigraph_template=mg,
        f_rel_update={
            # the input layer
            ("in", "cell"): GraphNet(
                f_inp=GraphNet.f_inp_concat,
                f_pool=GraphNet.f_pool_attn(
                    d_key=8, d_val=16, N_heads=8),
                f_v_up=GraphNet.f_v_up_beta(),
                f_e_up=GraphNet.f_e_up_dense(),
                f_adj_up=GraphNet._f_adj_up),
            # the working memory layer
            ("cell", "cell"): GraphNet(
                f_inp=GraphNet.f_inp_concat,
                f_pool=GraphNet.f_pool_attn(
                    d_key=16, d_val=64, N_heads=8),
                f_v_up=GraphNet.f_v_up_beta(),
                f_e_up=GraphNet.f_e_up_attn(
                    d_key=8, d_val=16, N_heads=8),
                f_adj_up=GraphNet._f_adj_up),
            # the output layer
            ("cell", "out"): GraphNet(
                f_inp=GraphNet.f_inp_concat,
                f_pool=GraphNet.f_pool_attn(
                    d_key=8, d_val=16, N_heads=8),
                f_v_up=GraphNet.f_v_up_beta(),
                f_e_up=GraphNet.f_e_up_attn(
                    d_key=4, d_val=8, N_heads=8),
                f_adj_up=GraphNet._f_adj_up)},
        f_inp=MultiGraphRNNCell.f_inp_update_root("in"),
        f_update_seq=(lambda x: [
            ("in", "cell"),
            ("cell", "cell"),
            ("cell", "out")]),
        f_ret=MultiGraphRNNCell.f_ret_just_root("out")))
    return rnn

dense2denseRNN(d_in=16, d_out=32)