from . import NodeOrgan, EdgeOrgan

import tensorflow as tf
keras = tf.keras
tfkl = keras.layers
K = keras.backend


class EnergyNode(NodeOrgan):

    def __init__(self, name, d_energy=8):

        super(EnergyNode, self).__init__(name=name, adaptive_potential=0)

        self._energy = K.zeros(shape=(d_energy,))

    def top_down(self):
        pass

    @property
    def get_free_energy(self):
        return K.sum(self.get_energy ** 0.5)

    @property
    def get_energy(self):
        return self._energy

    def set_energy(self, energy):
        # tf.assert_greater(energy, 0, 'energy components must be non-negative')
        self._energy = energy


class EnergyStorage(NodeOrgan):

    def __init__(self, energy_vecs, f_route, d_latent=8, **kwargs):
        super(EnergyStorage, self).__init__(**kwargs)

        self._energy_vecs = K.variable(energy_vecs)  # N_vecs x d_energy
        self._f_route = f_route  # R^n output

        self._latent_emb = K.random_uniform(d_latent)
        self._bound_energy = K.zeros(shape=self._energy_vecs.get_shape()[0])

    @property
    def get_info_state(self):
        return {
            'controllable': {
                'latent': {
                    'emb': self._latent_emb
                },
            },
            'uncontrollable': {
                'free_energy': self.get_free_energy,
                'bound_energy': self._bound_energy
            }
        }

    def bottom_up(self):
        free_energy_vec_coefficents = TODO ( self.energy_node.get_energy , self._energy_vecs )

        routing, self._new_latent, self._latent_weight = self._f_route(
            free_energy=free_energy_vec_coefficents,
            bound_energy=self._bound_energy,
            latent_emb=self._latent_emb
        )

        # here 'in' refers to flowing into the energy storage and
        # 'out' to energy leaving storage to the energy node for public use
        inflow_coef = K.relu(routing)
        outflow_coef = K.relu(-routing)
        attempted_inflow_vec = tf.matmul(a=inflow_coef,
                                         b=self._energy_vecs,
                                         transpose_a=True)
        attempted_outflow_vec = tf.matmul(a=outflow_coef,
                                          b=self._energy_vecs,
                                          transpose_a=True)
        available_free_energy = self.energy_node.get_energy
        available_bound_energy = tf.matmul(a=self._bound_energy,
                                           b=self._energy_vecs,
                                           transpose_a=True)
        actual_inflow_vec = K.minimum(available_free_energy, attempted_inflow_vec)
        actual_outflow_vec = K.minimum(available_bound_energy, attempted_outflow_vec)
        actual_net_influx = actual_inflow_vec - actual_outflow_vec

        self.energy_node.set_energy(available_free_energy - actual_net_influx)
        self._bound_energy += actual_net_influx

        self.set_free_energy(K.sum((actual_inflow_vec - attempted_inflow_vec) ** 2 +
                                   (actual_outflow_vec-attempted_outflow_vec) ** 2))

    def top_down(self):
        target_latents = [
            [entropy(target_latent), target_latent]  # TODO
            for target_latent in list(self._target_states.values())] + \
            [self._latent_weight, self._new_latent]
        normalized_weights = K.softmax(K.variable(target_latents[:, 0]))
        self._latent_emb = K.sum(normalized_weights * target_latents[:, 1])
        super(EnergyStorage, self).top_down()

    def _start_adaptation(self):
        raise NotImplementedError()  # TODO start async adaptation with reverb server data


class EnergyReactor(NodeOrgan):

    def __init__(self, energy_vecs, f_route, d_latent=8, **kwargs):
        super(EnergyReactor, self).__init__(**kwargs) # TODO I stopped at this line. Everything below needs work =============

        self._energy_vecs = K.variable(energy_vecs)  # N_vecs x d_energy
        self._f_route = f_route  # R^n output

        self._latent_emb = K.random_uniform(d_latent)
        self._bound_energy = K.zeros(shape=self._energy_vecs.get_shape()[0])

    @property
    def get_info_state(self):
        return {
            'controllable': {
                'latent': {
                    'emb': self._latent_emb
                },
            },
            'uncontrollable': {
                'free_energy': self.get_free_energy,
                'bound_energy': self._bound_energy
            }
        }

    def bottom_up(self):
        free_energy_vec_coefficents = TODO ( self.energy_node.get_energy , self._energy_vecs )

        routing, self._new_latent, self._latent_weight = self._f_route(
            free_energy=free_energy_vec_coefficents,
            bound_energy=self._bound_energy,
            latent_emb=self._latent_emb
        )

        # here 'in' refers to flowing into the energy storage and
        # 'out' to energy leaving storage to the energy node for public use
        inflow_coef = K.relu(routing)
        outflow_coef = K.relu(-routing)
        attempted_inflow_vec = tf.matmul(a=inflow_coef,
                                         b=self._energy_vecs,
                                         transpose_a=True)
        attempted_outflow_vec = tf.matmul(a=outflow_coef,
                                          b=self._energy_vecs,
                                          transpose_a=True)
        available_free_energy = self.energy_node.get_energy
        available_bound_energy = tf.matmul(a=self._bound_energy,
                                           b=self._energy_vecs,
                                           transpose_a=True)
        actual_inflow_vec = K.minimum(available_free_energy, attempted_inflow_vec)
        actual_outflow_vec = K.minimum(available_bound_energy, attempted_outflow_vec)
        actual_net_influx = actual_inflow_vec - actual_outflow_vec

        self.energy_node.set_energy(available_free_energy - actual_net_influx)
        self._bound_energy += actual_net_influx

        self.set_free_energy(K.sum((actual_inflow_vec - attempted_inflow_vec) ** 2 +
                                   (actual_outflow_vec-attempted_outflow_vec) ** 2))

    def top_down(self):
        target_latents = [
            [entropy(target_latent), target_latent]  # TODO
            for target_latent in list(self._target_states.values())] + \
            [self._latent_weight, self._new_latent]
        normalized_weights = K.softmax(K.variable(target_latents[:, 0]))
        self._latent_emb = K.sum(normalized_weights * target_latents[:, 1])
        super(EnergyStorage, self).top_down()

    def _start_adaptation(self):
        raise NotImplementedError()  # TODO start async adaptation with reverb server data