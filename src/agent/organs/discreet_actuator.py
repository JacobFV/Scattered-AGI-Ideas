from ..nodes import ActuatorNode, N_SAMPLES
from ..lca import DepthwiseConv2DLCA, SplitLCA

import tensorflow as tf
import tensorflow_probability as tfp

keras = tf.keras
K = keras.backend

tfd = tfp.distributions


class DiscreetActuator(ActuatorNode):

    def __init__(self, n, name=None):
        self.n = n
        super(DiscreetActuator, self).__init__(d_zc=n, d_zu=1, name=name)

    def get_action(self):
        self.zcs = self.child_targets[-1][-1].sample()
        self.child_targets.clear()

        indeces = K.argmax(self.zcs, axis=-1)
        action_index_dist = tfd.Empirical(indeces)
        action_index = action_index_dist.sample()
        self.zus = - action_index_dist.entropy() - action_index_dist.log_prob(action_index)
        return action_index