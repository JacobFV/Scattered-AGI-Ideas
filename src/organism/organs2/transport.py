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
        flowrate = self._f_fr(
            fe=self.get_free_energy,
            energies=self.src_energy_node.get_energy)
        # fr-vector: [d_energy]
        moving_energy = flowrate * K.relu(
            self.src_energy_node.get_energy - self.dst_energy_node.get_energy)
        self.src_energy_node.set_energy(self.src_energy_node.get_energy - moving_energy)
        self.dst_energy_node.set_energy(self.dst_energy_node.get_energy + moving_energy)

    def top_down(self):
        self.set_free_energy((self.src.get_free_energy -
                              self.dst.get_free_energy) ** 2)
        super(PassiveTransportChannel, self).top_down()

    def _start_adaptation(self):
        raise NotImplementedError() # TODO start async adaptation with reverb server data


class ActiveTransportChannel(EdgeOrgan):

    def __init__(self, energy_vec, f_flowrate, d_latent=8, **kwargs):
        super(ActiveTransportChannel, self).__init__(**kwargs)

        self._energy_vec = energy_vec # same dimensionality as energy
        self._f_flowrate = f_flowrate # should include bias terms,
        # last layer should use relu, elu, etc to ensure positive output
        self._latent_emb = K.random_uniform(d_latent)

    @property
    def get_info_state(self):
        """returns a dict like:
        { 'controllable': dict<str, val|dict>,
          'uncontrollable': dict<str, val|dict> }
        """
        return  {
            'controllable': {
                'latent': {
                    'emb': self._latent_emb
                },
            },
            'uncontrollable': {
                'free_energy': self.get_free_energy
            }
        }

    def bottom_up(self):
        energy_gradient = K.mean((self.src_energy_node.get_energy -
                                  self.dst_energy_node.get_energy) ** 2)
        # energy_gradient: scalar (,)
        policy_flowrate, new_latent, new_latent_weight = self._f_flowrate(
            energy=self.src_energy_node.get_energy,
            energy_gradient=energy_gradient,
            latent_emb=self._latent_emb)
        # fr: scalar (,)
        self.set_target_info_state(target_controllable_state=new_latent,
                                   target_weight=new_latent_weight,
                                   callee=self)

        demanded_energy = (policy_flowrate - energy_gradient) ** 2 # scalar
        available_energy = self.src_energy_node.get_energy VEC_DIV self._energy_vec # scalar
        # TODO find how many multiples of self._energy_vec are in src's energy
        missing_pump_energy = K.relu(available_energy - demanded_energy) # scalar
        consumed_energy = demanded_energy - missing_pump_energy # scalar
        attempted_flowrate = (consumed_energy ** 0.5) + energy_gradient # scalar

        # now transport the energy from src to dst
        component_flowrates = K.minimum(self.src_energy_node.get_energy, attempted_flowrate)
        moving_energy = component_flowrates * self.src_energy_node.get_energy
        self.src_energy_node.set_energy(self.src_energy_node.get_energy - moving_energy)
        self.dst_energy_node.set_energy(self.dst_energy_node.get_energy + moving_energy)

        # assign free energy to state
        self.set_free_energy(missing_pump_energy + (attempted_flowrate - component_flowrates) ** 2)

    def top_down(self):
        normalized_weights = K.softmax(K.variable(list(self._target_states.values())[:,0]))
        self._latent_emb = K.sum(normalized_weights * K.variable([
            d['latent']['emb'] for d in list(self._target_states.values())[:,1]]))

        super(ActiveTransportChannel, self).top_down()

    def _start_adaptation(self):
        raise NotImplementedError() # TODO start async adaptation with reverb server data