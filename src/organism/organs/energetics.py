from . import NodeOrgan
from .brain.nn.som import SOM

import tensorflow as tf
keras = tf.keras
K = keras.backend
import gym


class EnergyNode(NodeOrgan):
    def __init__(self, d_energy, receptor_vecs, max_energy, **kwargs):
        super(EnergyNode, self).__init__(**kwargs)

        self.d_energy = d_energy
        self.receptor_vecs = receptor_vecs
        self.max_energy = max_energy

        self.energy = tf.zeros((d_energy,))
        self.health = tf.constant(1.0)

        self.SOM = SOM(initial_vecs=receptor_vecs)

    def get_observation_space(self):
        return gym.spaces.Box(low=0, high=1., shape=len(self.receptor_vecs))

    def get_observation(self):
        return self.SOM(self.energy, wta=False)

    def step(self):
        self.health -= K.log(1 + K.relu(K.sum(self.energy) - self.max_energy))
        self.energy *= self.health

    def train(self):
        self.health = K.sigmoid(self.health + 0.1)


class EnergyConverter(NodeOrgan):
    def __init__(self, d_energy, receptor_vecs, d_latent=4, **kwargs):
        super(EnergyConverter, self).__init__(**kwargs)

        self.d_energy = d_energy
        self.receptor_vecs = receptor_vecs
        self.d_latent = d_latent

        self.latent = tf.random.uniform((self.d_latent))

        self.SOM = SOM(initial_vecs=receptor_vecs)
        

    def get_observation_space(self):
        return gym.spaces.Box(low=0, high=1., shape=len(self.d_latent))

    def get_observation(self):
        return self.latent

    def get_action_space(self):
        return gym.spaces.Box(low=0, high=1., shape=len(self.d_latent))

    def set_action(self, action):
        beta = self.node.health
        self.latent = beta * action + (1-beta) * self.latent

    def step(self):
        energy_comps = self.SOM
        consume_comps =
        produce_comps =

    def train(self, training_level):
        if training_level > 3:
            return
        self.SOM

#converter
#transfer
#storage
