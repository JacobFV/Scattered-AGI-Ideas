from .. import NodeOrgan

import tensorflow as tf
keras = tf.keras
tfkl = keras.layers
K = keras.backend


class InformationNode(NodeOrgan):

    def __init__(self, f_abs, f_pred, f_act, d_rel=8, **kwargs):
        super(InformationNode, self).__init__(**kwargs)

        self.f_abs = f_abs
        self.f_pred = f_pred
        self.f_act = f_act

        self._neighbors = dict() # dict<neighbor:InformationNode, latent translator:callable>

        self.d_rel = d_rel

        self._latent = {k: tf.zeros(shape)
                        for k, shape
                        in self.get_controllable_state_space['controllable']['latent'].items()}
        self._pred_latent = {k: tf.zeros(shape)
                        for k, shape
                        in self.get_controllable_state_space['controllable']['latent'].items()}

    def add_neighbor(self, neighbor, translator):
        self._neighbors[neighbor] = translator

    def bottom_up(self):
        self._latent = self.f_abs(latent=self._latent,
                                  parent_states=self._get_parent_states.values())
        # the latent should guide querying that way high entropy latents
        # are more responsive to low entropy stimuli
        # self._latent: 1-Tensor, 2-Tensor, any...

        self._set_fe(sum([
            KL( latent_component, prediction_component ) # TODO
            for latent_component, prediction_component
            in zip(list(self._latent.values()),
                   list(self._pred_latent.values()))
        ]))

        self.pred_latent = self.f_pred(latent=self._latent)
        # self._pred_latent: `self._latent`-like

    def top_down(self):
        # give more weight to salient 'free-energy' stimuli and less
        # to entropic, uncertain stimuli. These are not exclusive.
        weighted_neighbor_predictions = [
            [
                neighbor.get_state['fe'] - entropy(neighbor.pred_latent), # TODO
                f_trans(neighbor.get_state)
            ]
            for neighbor, f_trans
            in self._neighbors.items()
        ]
        weighted_child_targets = [
            [
                child.get_state['fe'] - entropy(target_latent), # TODO
                target_latent
            ]
            for child, target_latent
            in self.child_targets
        ]
        combined_influences = \
            weighted_neighbor_predictions + \
            weighted_child_targets + [[
                self.get_state['fe'] - entropy(self.pred_latent), # TODO
                self.pred_latent
            ]]

        normalized_weights = K.softmax(K.stack(combined_influences[:,0], axis=0))

        target_latent = {
            k: sum(normalized_weights * combined_influences[:,1][k])
            for k, _ in self.get_state_space['controllable']['latent'].items()
        }

        target_parent_states = self.f_act(
            latent=self._latent,
            target=target_latent,
            current_parents=self._get_parent_states)

        for parent, target_state in target_parent_states:
            parent.set_target_state(target_state=target_state, callee=self)

        super(InformationNode, self).top_down()\

    @property
    def _get_parent_states(self):
        parents = {
            parent: (parent.get_state, rel)
            for rel, parent in self._parents.items()
        }
        for parent, (state, rel) in parents.items():
            parents[parent]['uncontrollable'].update({'rel': rel})
        return parents

    @property
    def get_state(self):
        # ordinary InformationNodes are almost fully controllable
        return {
            'controllable': {
                'latent': self._latent,
            },
            'uncontrollable': {
                'fe': self._get_fe,
            }
        }

    @property
    def get_state_space(self):
        """this method should be overridden for non-linear-structured latents"""
        return {
            'controllable': {
                'latent': {
                    'emb': (32,),
                },
            },
            'uncontrollable': {
                'fe': (1,),
            },
        }
