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

        perception_energy = self._latent_dist.kl_divergence(self.pred_latent_dist)
        self.pred_latent_dist = BIJECT(self._latent_dist,
                                       lambda latent: self.f_pred(latent=latent)) # joint Distribution
        # self.pred_latent_dist: `self._latent`-like
        self.pred_latent = self.pred_latent_dist.sample()
        self.pred_latent_energy = self.pred_latent_dist.logp(self.pred_latent) # TODO: + logp or - logp
        self.pred_latent_dist_energy = -self.pred_latent_dist.entropy()

        self.set_free_energy(perception_energy + self.pred_latent_energy + self.pred_latent_dist_energy)

    def top_down(self):
        # give more weight to salient 'free-energy' stimuli and less
        # to entropic, uncertain stimuli. These are not exclusive.

        for neighbor, f_trans in self._neighbors.items():
            self.set_target_info_state(
                target_controllable_state=f_trans(neighbor.pred_latent),
                weight=neighbor.get_free_energy,
                callee=neighbor)

        self.set_target_info_state(
            target_controllable_state=self.pred_latent,
            weight=self.get_free_energy,
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

        for parent, (target_state_sample, target_state_weight) in target_parent_states:
            parent.set_target_info_state(
                target_controllable_state=target_state_sample,
                weight=target_state_weight,
                callee=self)

        super(InformationNode, self).top_down()

    def _start_adaptation(self):


        reverb_client = None
        batch_size = 8

        # select memories to train on
        num_episodes = f(self._rolling_free_energy)

        train_batch = list()
        t_since_last_train = 0
        building_buffer = False
        for experience in REVERB_BUFFER:
            t_since_last_train += 1
            if experience.rolling_free_energy > X:
                building_buffer = True
            elif not building_buffer:
                continue

            if building_buffer:
                fv

        # train in batches
        def gen_batch(): pass

        def train_on_batch(): pass

            # train f_{abs} with dynamic forward propagation
            # unsupervised learning (if enabled in model)
            # _ = f_{abs} ( x^u, x^c, z_{t-1} ; training=True )

            # train f_{abs} and f_{act} together by targeting a predicted latent
            # min_{f_{act}, f_{abs}} KL [ Z_{t+1} || Z^{pred}_{t+1} ]
            # stopgradient before Z_t.
            # maybe even load Z_t, z^{*,comb}_{t+1}, x^u_{t+1} from buffer
            # x^c_{t+1} = f_{act} ( z^{*,comb}_{t+1} )
            # Z_{t+1} = f_{abs} ( Z_t , x^u_{t+1} , x^c_{t+1} )

            # train f_pred by targeting the true next latent
            # min_{f_{pred}} KL KL [ Z^{pred}_{t+1} || Z_{t+1} ]
            # stopgradient before Z_t and Z_{t+1}
            # maybe even load Z_t, Z_{t+1} from buffer
            # Z^{pred}_{t+1} = f_pred(Z_t)

            # train f_trans by association

            pass
        pass

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
