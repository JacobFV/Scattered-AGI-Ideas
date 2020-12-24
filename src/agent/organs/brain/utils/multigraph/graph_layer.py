import tensorflow as tf
keras = tf.keras
tfkl = keras.layers


class GraphLayer(keras.Model):

    def __init__(self,
                 f_inp,
                 f_pool,
                 f_v_up,
                 f_e_up,
                 f_adj_up,
                 **kwargs):
        """
        """

        super(GraphLayer, self).__init__(**kwargs)

        self.f_inp = f_inp
        self.f_pool = f_pool
        self.f_v_up = f_v_up
        self.f_e_up = f_e_up
        self.f_adj_up = f_adj_up

        self.f_V_src_loc = tfkl.Lambda(lambda inps: # V_src, A = inps
                                       tf.einsum('...sv,...sd->...sdv', inps[0], inps[1]))
        self.f_V_dst_loc = tfkl.Lambda(lambda inps: # V_dst, A = inps
                                       tf.einsum('...dv,...sd->...dsv', inps[0], inps[1]))
        self.f_perm = tfkl.Lambda(lambda x: tf.einsum('...sdv->...dsv', x))

    def call(self, inputs, training=False):
        """
        """

        # unpack inputs
        V_src, V_dst, E, A = inputs

        # provide vert-localized copies of src and dst verts
        V_src_loc = self.f_V_src_loc([V_src, A])
        V_dst_loc = self.f_V_dst_loc([V_dst, A])

        # get src-dst pair-specific inputs to dst verts

        print('layer:', self.name,
              'V_src', V_src.shape,
              'V_dst', V_dst.shape,
              'V_src_loc', V_src_loc.shape,
              'E', E.shape, 'A', A.shape)
        inp = self.f_inp([V_src_loc, E]) #, training=training)
        inp = self.f_perm(inp)
        # now `inp.shape` == (... dst, src, val)

        # pool src-dst pair-specific inputs
        V_dst_new = self.f_pool([V_dst, inp]) #, training=training)

        # update dst verts
        V_dst = self.f_v_up([V_dst, V_dst_new]) #, training=training)

        # update edges
        E = self.f_e_up([V_src_loc, V_dst_loc, E]) #, training=training)
        # [..., N_src, N_dst, d_e]

        # update adjacency matrix
        A = self.f_adj_up(E) #, training=training)
        # [..., N_src, N_dst]

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
            super(GraphLayer.f_pool_attn, self).__init__(**kwargs)

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

            self.reshape_q = tfkl.Reshape(V_dst_shape[1:-1] +
                                          (self.N_heads, self.d_key))
            self.reshape_k = tfkl.Reshape(inp_shape[1:-1] +
                                          (self.N_heads, self.d_key))
            self.reshape_v = tfkl.Reshape(inp_shape[1:-1] +
                                          (self.N_heads, self.d_val))

            def _f_MHA(inps):
                queries, keys, values = inps
                score = tf.einsum('...dhq,...dshq->...dsh', queries, keys)
                score = score / self.d_key**0.5
                score = tf.nn.softmax(score, axis=-1)
                return tf.einsum('...dsh,...dshv->...dhv', score, values)

            self.f_MHA = tfkl.Lambda(lambda x: _f_MHA(x))

            self.f_cat = tfkl.Reshape(V_dst_shape[1:-1] + (self.N_heads*self.d_val,))
            self.f_emb_cat = tfkl.Dense(V_dst_shape[-1], 'relu')

        def call(self, inputs, training=False):
            # unpack inputs
            V_dst, inp = inputs
            # inp: [B, N_dst, N_src, d_e+d_vsrc]
            # V_dst: [B, N_dst, d_vdst]

            # pre-LN
            if self.pre_layer_normalization:
                V_dst = self.V_dst_LN(V_dst) #, training=training)
                inp = self.inp_LN(inp) #, training=training)
                # in2cell   (4, 64, 1, 25)  dv:24 de:1 | (4, 64, 1, 28) dv:24 de:4
                # cell2cell (4, 64, 64, 24) dv:16 de:8 |
                # cell2out  (4, 1, 64, 17)  dv:16 de:1 |

            # generate queries, keys, and values for all heads
            queries = self.f_query(V_dst) #, training=training)  # [..., N_dst, N_heads*d_key]
            keys = self.f_key(inp) #, training=training)  # [..., N_dst, N_src, N_heads*d_key]
            values = self.f_val(inp) #, training=training)  # [..., N_dst, N_src, N_heads*d_val]

            # reshape into separate heads
            queries = self.reshape_q(queries)  # [..., N_dst, N_heads, d_key]

            keys = self.reshape_k(keys)  # [..., N_dst, N_heads, d_key]
            values = self.reshape_v(values)  # [..., N_dst, N_heads, d_key]

            # perform multi-head attention
            mha_lookup = self.f_MHA([queries, keys, values], training=training)
            # [..., N_dst, N_heads, d_val]

            # concatenate heads
            mha_cat = self.f_cat(mha_lookup) #, training=training)
            # [..., N_dst, N_heads*d_val]

            # embed in output space
            return self.f_emb_cat(mha_cat) #, training=training)

    @staticmethod
    def f_v_up_add():
        return tfkl.Add()

    @staticmethod
    def f_v_up_direct():
        return tfkl.Lambda(lambda V_dst, V_dst_new: V_dst_new)

    class f_v_up_beta(tfkl.Layer):
        def __init__(self, **kwargs):
            super(GraphLayer.f_v_up_beta, self).__init__(**kwargs)
            self.f_beta = tfkl.Dense(1, 'softmax')

        def call(self, inputs, training=False):
            V_dst, V_dst_new = inputs
            beta = self.f_beta(V_dst_new)
            return beta * V_dst + (1 - beta) * V_dst_new

    class f_v_up_alphabeta(tfkl.Layer):
        def __init__(self, **kwargs):
            super(GraphLayer.f_v_up_alphabeta, self).__init__(**kwargs)
            self.f_beta = tfkl.Dense(1, 'softmax')
            self.f_alpha = tfkl.Dense(1, 'softmax')

        def call(self, inputs, training=False):
            V_dst, V_dst_new = inputs
            alpha = self.f_alpha(V_dst)
            beta = self.f_beta(V_dst_new)
            return alpha * V_dst + beta * V_dst_new

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
        dense_layer = tfkl.Dense(1, 'softmax')
        lambda_layer = tfkl.Lambda(lambda E: tf.squeeze(E, axis=-1))
        def call_adj_up(x):
            x = dense_layer(x)
            return lambda_layer(x)
        return call_adj_up

    @staticmethod
    def f_e_up_const():
        return tfkl.Lambda(lambda V_src_loc, V_dst_loc, E: E)

    class f_e_up_dense(tfkl.Layer):
        def __init__(self, **kwargs):
            super(GraphLayer.f_e_up_dense, self).__init__(**kwargs)

        def build(self, input_shape):
            V_src_loc_shape, V_dst_loc_shape, E_shape = input_shape
            _shp = E_shape[-1]
            self.f_E_new = tfkl.Dense(E_shape[-1], 'relu')
            self.V_dst_perm = tfkl.Lambda(
                lambda x: tf.einsum('...dsv->...sdv', x))

        def call(self, inputs, training=False):
            V_src_loc, V_dst_loc, E = inputs
            V_dst_loc_perm = self.V_dst_perm(V_dst_loc)
            return self.f_E_new(tfkl.concatenate([
                V_src_loc, V_dst_loc_perm, E]))

    class f_e_up_dense_oneway(tfkl.Layer):
        def __init__(self, **kwargs):
            super(GraphLayer.f_e_up_dense_oneway, self).__init__(**kwargs)

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
            super(GraphLayer.f_e_up_beta, self).__init__(**kwargs)

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
            return beta * V_dst_loc_perm + (1 - beta) * E_new

    class f_e_up_attn(tfkl.Layer):

        def __init__(self, d_key=8, d_val=None, N_heads=8, pre_layer_normalization=True, **kwargs):
            """
            pre-LN (https://arxiv.org/abs/2004.08249)
            """
            super(GraphLayer.f_e_up_attn, self).__init__(**kwargs)

            self.pre_layer_normalization = pre_layer_normalization
            self.d_key = d_key
            self.d_val = self.d_key if d_val is None else d_val
            self.N_heads = N_heads

            if self.pre_layer_normalization:
                self.V_dst_LN = tfkl.LayerNormalization()
                self.inp_LN = tfkl.LayerNormalization()
                self.E_LN = tfkl.LayerNormalization()

        def build(self, input_shape):
            V_src_loc_shape, V_dst_loc_shape, E_shape = input_shape

            self.V_dst_perm = tfkl.Lambda(
                lambda x: tf.einsum('...dsv->...sdv', x))

            self.cat_q_data = tfkl.Concatenate()
            self.cat_kv_data = tfkl.Concatenate()

            self.f_query = tfkl.Dense(self.d_key, 'relu')

            self.f_key = tfkl.Dense(self.N_heads * self.d_key, 'relu')
            self.f_val = tfkl.Dense(self.N_heads * self.d_val, 'relu')

            self.reshape_k = tfkl.Reshape(E_shape[1:-1] +
                                          (self.N_heads, self.d_key))
            self.reshape_v = tfkl.Reshape(E_shape[1:-1] +
                                          (self.N_heads, self.d_val))

            def _f_MHA(inps):
                queries, keys, values = inps
                score = tf.einsum('...sdq,...sdhq->...sdh', queries, keys)
                score = score / self.d_key**0.5
                score = tf.nn.softmax(score, axis=-1)
                return tf.einsum('...sdh,...sdhv->...sdv', score, values)

            self.f_MHA = tfkl.Lambda(lambda x: _f_MHA(x))

            self.f_cat = tfkl.Reshape(E_shape[1:-1] + (self.N_heads*self.d_val,))
            # Input to reshape is a tensor with 32768 values, but the requested shape has 2097152 [Op:Reshape]
            self._shp = E_shape[1:-1] + (self.N_heads*self.d_val,) # (64,64,128)
            self.f_emb_cat = tfkl.Dense(E_shape[-1], 'relu')

        def call(self, inputs, training=False):
            # unpack inputs
            V_src_loc, V_dst_loc, E = inputs

            # pre-LN
            if self.pre_layer_normalization:
                V_dst_loc = self.V_dst_LN(V_dst_loc) #, training=training)
                V_src_loc = self.inp_LN(V_src_loc) #, training=training)
                E = self.E_LN(E) #, training=training)

            V_dst_loc_perm = self.V_dst_perm(V_dst_loc) # [..., N_src, N_dst, d_v]

            q_data = self.cat_q_data([V_dst_loc_perm, E]) # [..., N_src, N_dst, d_v_dst + d_e]
            kv_data = self.cat_kv_data([V_src_loc, E]) # [..., N_src, N_dst, d_v_src + d_e]

            # generate queries
            queries = self.f_query(q_data) #, training=training)  # [..., N_src, N_dst, d_key]
            # [..., N_src, N_dst, N_heads, d_key]

            # generate multiple keys and values
            keys = self.f_key(kv_data) #, training=training)  # [..., N_src, N_dst, N_heads*d_key]
            values = self.f_val(kv_data) #, training=training)  # [..., N_src, N_dst, N_heads*d_val]

            # reshape into separate heads
            keys = self.reshape_k(keys)  # [..., N_src, N_dst, N_heads, d_key]
            values = self.reshape_v(values)  # [..., N_src, N_dst, N_heads, d_key]

            # perform multi-head attention
            mha_lookup = self.f_MHA([queries, keys, values]) #, training=training)
            # [..., N_src, N_dst, d_val]

            # embed in output space
            return self.f_emb_cat(mha_lookup) #, training=training)
            # [..., N_src, N_dst, d_E]