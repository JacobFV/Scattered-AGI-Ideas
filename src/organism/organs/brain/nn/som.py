import tensorflow as tf
keras = tf.keras
tfkl = keras.layers
K = keras.backend

class SOM(tfkl.Layer):
    """
    c = xM
    x: [b, n_x]
    M: [n_x, n_y]
    c: [b, n_y]

    M^-1: [n_y, n_x]
       [ -v1- ]
     = [ -v2- ]
       [ ...  ]

    let M^-1 = N
    N^-1 = M
    """

    def __init__(self, initial_vecs, lr=1e-3):
        self.lr = lr
        self.change_of_basis_matrix = tf.linalg.pinv(tf.constant(initial_vecs))

    def call(self, inputs, training=True, wta=True):
        components = inputs @ self.change_of_basis_matrix
        if wta:
            # apply a hard (non-differentiable) winner-take-all coding filter
            components *= K.one_hot(indices=K.argmax(components), num_classes=components.shape[-1])
        if training:
            # TODO perform batch-wise winner take all and compute increments
            raise NotImplementedError()
        return components
