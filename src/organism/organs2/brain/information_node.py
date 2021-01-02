from .. import NodeOrgan
from ... import utils

import tensorflow as tf
import tensorflow_probability as tfp

keras = tf.keras
tfkl = keras.layers
K = keras.backend

tfb = tfp.bijectors
tfpl = tfp.layers
tfd = tfp.distributions


class InformationNode(NodeOrgan):

    def __init__(self,
                 f_abs,
                 f_pred,
                 f_act,
                 latent_structure, # eg: {"emb": (16,)} or {"V": (10,8), "E": (10,10,4)}
                 d_rel=8,
                 **kwargs):
        super(InformationNode, self).__init__(**kwargs)

        self.f_abs = f_abs
        self.f_pred = f_pred
        self.f_act = f_act

        self.children = list() # list<_:InformationNode>
        self._parents = dict() # dict<relation:InformationNode>
        self._neighbors = dict() # dict<neighbor:InformationNode, latent translator:callable>

        self.d_rel = d_rel

        self._latent_dist = tfd.JointDistributionNamed(model=utils.structured_op(
            latent_structure, lambda shape: tfd.Uniform(high=tf.ones(shape=shape))))
        self._latent = self._latent_dist.sample()
        self.pred_latent_dist = self._latent_dist

    def add_neighbor(self, neighbor, translator):
        self._neighbors[neighbor] = translator

    def bottom_up(self):
        self._latent_dist = BIJECT(self._latent_dist,
                              lambda latent: self.f_abs(latent=latent,
                                  parent_states=self._get_parent_states.values()))  # dict Distribution
        # the latent should guide querying that way high entropy latents
        # are more responsive to low entropy stimuli
        self._latent = self._latent_dist.sample()

        self.set_free_energy(self._latent_dist.kl_divergence(self.pred_latent_dist))
            #utils.reduce_sum_dict(utils.pairwise_structured_op(
            #self._latent_dist, self.pred_latent_dist,
            #lambda true_dist, target_dist: true_dist.kl_divergence(target_dist))))

        self.pred_latent_dist = BIJECT(self._latent_dist,
                                       lambda latent: self.f_pred(latent=latent)) # joint Distribution
        # self.pred_latent_dist: `self._latent`-like

    def top_down(self):
        # give more weight to salient 'free-energy' stimuli and less
        # to entropic, uncertain stimuli. These are not exclusive.

        for neighbor, f_trans in self._neighbors.items():
            self.set_target_info_state(
                target_controllable_state=f_trans(neighbor.pred_latent_dist),
                weight=neighbor.get_free_energy - utils.reduce_sum_dict(
                   utils.structured_op(neighbor.pred_latent_dist, lambda x: x.entropy())),
                callee=neighbor)
        # TODO stopped here
        # I need to make the `set_target_info_state convert any tensors into point distributions.
        # this may not be possible so I will need to otherwise biject the self and neighbor predicted
        # latent distributions (joined as one `Independant Joint` distribution) into the parents targets
        # finally, take a sample and measure log_prob and entopy of sample and dist to set parent targets
        # TODO I also haven't finished naming latent to _latent_dist

        self.set_target_info_state(
            target_controllable_state=self.pred_latent_dist,
            weight=self.get_free_energy - self.pred_latent_dist.entropy(),
            callee=self)

        normalized_weights = K.softmax(K.stack(list(self.child_targets.values())[:, 0], axis=0))

        target_latent = {
            k: sum(normalized_weights * list(self.child_targets.values())[:, 1][k])
            for k, _ in self.get_info_state_space['controllable']['latent'].items()
        }

        target_parent_states = self.f_act(
            latent=self._latent,
            target=target_latent,
            current_parents=self._get_parent_states)

        for parent, (target_state_sample, target_state_entropy) in target_parent_states:
            parent.set_target_info_state(
                target_controllable_state=target_state_sample,
                weight=target_state_entropy,
                callee=self)

        super(InformationNode, self).top_down()

    @property
    def _get_parent_states(self):
        parent_vals = dict()
        for rel, parent in self._parents.items():
            parent_vals[parent] = parent.get_info_state
            parent_vals[parent]['uncontrollable']['rel'] = rel
        return parent_vals

    @property
    def get_info_state(self):
        # ordinary InformationNodes are almost fully controllable
        return {
            'controllable': {
                'latent': self._latent,
            },
            'uncontrollable': {
                'free_energy': self.get_free_energy,
            }
        }
