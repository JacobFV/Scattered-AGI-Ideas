import tensorflow as tf
keras = tf.keras
tfkl = keras.layers
K = keras.backend
import gym


class InformationNode:

    def get_controllable_state_space(self): # not `observation_space`
        raise NotImplementedError()

    def get_uncontrollable_prestate_space(self): #
        """
        `pre` because it has yet to be tuple appended with the parent relationship
        in heirarchial settings xu = xu+(rel,)
        xu is always a tuple and free energy is the last entry (before appending `rel`, then it's second-last)
        not `observation_space`
        """
        raise NotImplementedError()


    def get_state(self):
        raise NotImplementedError()

    def set_target_state(self, target_state, callee): # called by others; rarely by self
        raise NotImplementedError()

    def set_fe(self, fe):
        self._fe = fe

    def get_fe(self):
        return self._fe

class PredictiveNode(InformationNode):

    # TODO: make rel_emb a part of the uncontrollable observation of each parent

    class LatentTranslator(keras.Model):

        def __init__(self, d_in, d_out, **kwargs):
            super(PredictiveNode.LatentTranslator, self).__init__(**kwargs)

            self.d_in = d_in
            self.d_out = d_out

        def build(self, input_shape):
            self.dense1 = tfkl.Dense(int(1.5*self.d_in), activation='relu', input_shape=input_shape[-1])
            self.dense2 = tfkl.Dense(int(self.d_in+self.d_out), activation='relu', input_shape=input_shape[-1])
            self.dense3 = tfkl.Dense(int(1.5*self.d_out), activation='relu', input_shape=input_shape[-1])
            self.dense4 = tfkl.Dense(self.d_out, activation='relu', input_shape=input_shape[-1])

        def call(self, inputs, training=None, mask=None):
            x = self.dense1(inputs)
            x = self.dense2(x)
            x = self.dense3(x)
            x = self.dense4(x)
            return x


    def __init__(self, f_abs, f_pred, f_act, d_rel=4):
        self.f_abs = f_abs
        self.f_pred = f_pred
        self.f_act = f_act

        self.d_rel = d_rel

        self.children = dict() # dict<<InformationNode, rel>
        self.parents = dict() # dict<<PredictiveNode, rel>
        self._neighbors = dict() # dict<PredictiveNode, f_translator>

        self._latent = tf.zeros(self.get_combined_state_space()[-1])
        self._pred_state = tf.zeros_like(self._latent)

        self.child_targets = dict() # dict<PredictiveNode, target_controllable_state>

    def add_neighbor(self, neighbor):
        self._neighbors[neighbor] = PredictiveNode.LatentTranslator(
            d_in=self.get_combined_state_space()['latent'],
            d_out=neighbor.get_combined_state_space()['latent'])

    def get_neighbors(self):
        return list(self._neighbors.keys())

    def bottom_up(self):
        observation = self._get_observation()
        # observation: dict<str, tuple<rel: 1-Tensor, xc: ?-Tensor, xu: ?-Tensor>>
        self._latent = self.f_abs(self._latent, observation)
        # self._latent: 1-Tensor
        self.set_fe(KL (self._latent, self._pred_latent)) # TODO
        # self._free_energy: 1-Tensor
        self._pred_latent = self.f_pred(self._latent)
        # self.pred_latent_dist: 1-Tensor

    def top_down(self):

        """I don't know if I should give more weight or less to more free energy"""
        neighbors_pred_latent_weights = list()
        translated_neighbor_latent = list()
        for neighbor, f_trans in self.get_neighbors().items():
            fe_weighted_entropy = neighbor._free_energy - entropy(neighbor.pred_latent_dist) # TODO
            neighbors_pred_latent_weights.append(fe_weighted_entropy)
            translated_neighbor_latent.append(f_trans(neighbor.pred_latent_dist))

        children_target_latent_weights = list()
        children_latents = list()
        for child, target_latent in self.child_targets:
            children_target_latent_weights.append(child._free_energy - entropy(target_latent)) # TODO
            children_latents.append(target_latent)

        combined_latent_weights = neighbors_pred_latent_weights + \
                                  children_target_latent_weights + \
                                  [self._fe - entropy(self._pred_latent)]
        combined_latents = translated_neighbor_latent + children_latents + [self._pred_latent]


    def _get_observation(self):
        parents = dict()
        for parent, rel in self.parents.items():
            state = parent.get_info_state()
            state['uncontrollable']['rel'] = rel
            parents[parent] = state
        return parents

    def _set_controllable_observation(self, target_xc_dict):
        """
        target_xc_dict (dict<InformationNode, xc>)
        """
        for parent, target in target_xc_dict.items():
            parent.set_target_info_state(target_controllable_state=target, callee=self)

    def get_state(self):
        # ordinary PredictiveNervousNodes are almost fully controllable
        return {'latent': self._latent, 'free_energy': self._fe}

    def set_target_state(self, target_latent, callee): # called by children; rarely by self
        self.child_targets[callee] = target_latent

    def get_controllable_state_space(self): # not `observation_space`
        return {'latent': self._latent.shape}

    def get_uncontrollable_state_space(self): # not `observation_space`
        return {'free_energy': (1,)}

    def get_combined_state_space(self):
        return self.get_controllable_state_space()\
            .update(self.get_uncontrollable_state_space())
