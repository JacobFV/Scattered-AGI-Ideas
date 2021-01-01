from . import NodeOrgan, EdgeOrgan

import tensorflow as tf
keras = tf.keras
tfkl = keras.layers
K = keras.backend


class PassiveTransportChannel(EdgeOrgan):

    def __init__(self, f_fr, d_energy=8, **kwargs):
        super(PassiveTransportChannel, self).__init__(**kwargs)

        self._f_fr = f_fr # should include bias terms,
        # last layer should use sigmoid to ensure (0,1) output

    def bottom_up(self):
        fr = self._f_fr(
            fe=self.get_state['uncontrollable']['fe'],
            energies=self.src_energy_node.get_energy)
        # fr-vector: [d_energy]
        moving_energy = fr * K.relu(
            self.src_energy_node.get_energy - self.dst_energy_node.get_energy)
        self.src_energy_node.set_energy(self.src_energy_node.get_energy - moving_energy)
        self.dst_energy_node.set_energy(self.dst_energy_node.get_energy + moving_energy)

    def top_down(self):
        self._set_fe((self.src.get_state['uncontrollable']['fe'] -
                      self.dst.get_state['uncontrollable']['fe']) ** 2)
        super(PassiveTransportChannel, self).top_down()

    def _start_adaptation(self):
        raise NotImplementedError() # TODO start async adaptation with reverb server data


class ActiveTransportChannel(EdgeOrgan):

    def __init__(self, energy_vec, f_fr, d_latent=8, **kwargs):
        super(ActiveTransportChannel, self).__init__(**kwargs)

        self._energy_vec = energy_vec # same dimensionality as energy
        self._f_fr = f_fr # should include bias terms,
        # last layer should use relu, elu, etc to ensure positive output

        self._state = {
            'controllable': {
                'latent': {
                    'emb': K.random_uniform(d_latent)
                },
            },
            'uncontrollable': {
                'fe': self._get_fe
            }
        }

    def bottom_up(self):
        energy_gradient = K.mean((self.src_energy_node.get_energy -
                                  self.dst_energy_node.get_energy) ** 2)
        # energy_gradient: scalar (,)
        fr, self._new_latent, self._latent_weight = self._f_fr(
            energy=self.src_energy_node.get_energy,
            energy_gradient=energy_gradient,
            latent=self.get_state['controllable']['latent']['emb'])
        # fr: scalar (,)

        demanded_energy = (fr - energy_gradient) ** 2 # scalar
        available_energy = self.src_energy_node.get_energy VEC_DIV self._energy_vec # scalar
        # TODO find how many multiples of self._energy_vec are in src's energy
        missing_pump_energy = K.relu(available_energy - demanded_energy) # scalar
        consumed_energy = demanded_energy - missing_energy # scalar
        attempted_fr = (consumed_energy ** 0.5) + energy_gradient # scalar

        # now transport the energy from src to dst
        component_flowrates = K.min(self.src_energy_node.get_energy, attempted_fr)
        moving_energy = component_flowrates * self.src_energy_node.get_energy
        self.src_energy_node.set_energy(self.src_energy_node.get_energy - moving_energy)
        self.dst_energy_node.set_energy(self.dst_energy_node.get_energy + moving_energy)

        # assign free energy to state
        self._set_fe(missing_pump_energy + (attempted_fr - component_flowrates) ** 2)

    def top_down(self):
        target_latents = [
            [ entropy(target_latent), target_latent ] # TODO
            for target_latent
            in list(self._target_states.values())] + \
            [ self._latent_weight, self._new_latent ]
        normalized_weights = softmax ( target_latents[:,0] ) # TODO
        self._state['controllable']['latent']['emb'] = K.sum(
            normalized_weights * target_latents[:,1])

        super(ActiveTransportChannel, self).top_down()

    def _start_adaptation(self):
        raise NotImplementedError() # TODO start async adaptation with reverb server data