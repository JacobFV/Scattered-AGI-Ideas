from .contact import NodeContact

import requests
import tensorflow as tf
keras = tf.keras
tfkl = keras.layers
import gym

class Intake(NodeContact):
    """this is not just a tongue"""

    def __init__(self, **kwargs):
        super(Intake, self).__init__(**kwargs)
        self.stomach = 0

    def get_observation_space(self):
        return gym.spaces.Tuple((
            gym.spaces.Box(low=0., high=1., shape=(self.d_emb,)), # embedding from contact and consumtion
            gym.spaces.Box(low=0., high=1., shape=(self.organism.d_energy,)) # pure embedding of food/air
                # before including the energy node. This is only activated when eating
        ))

    def get_action_space(self):
        # [pitch, yaw, dof, focal_length]
        return gym.spaces.Box(low=0., high=1., shape=(1,))

    def consume_step(self, feel_tags, consume_tags):
        tag_emb = tf.reduce_sum(self.emb_fn(tags), axis=0) # TODO

        difference = self.node.energy - self.stomach.energy # TODO
        self.node.energy = self.node.energy + difference * # TODO


class UnityIntake(Intake):

    def step(self):
        # actually get objects in contact
        r = requests.get(
            f'{self.organism.env_comm.ip}/try_consuming_objs_in_contact',
            params={
                'name':f'{self.organism.get_name}_{self.get_name}',
                'consumable_type': self.contact_type})
        unity_names, tags = r.json()
        if self.action[0] > 0.5:
            requests.get(
                f'{self.organism.env_comm.ip}/remove_objs',
                params={'names': unity_names}
            )
            super(UnityIntake, self).consume_step(feel_tags=tags, consume_tags=tags)
        else:
            super(UnityIntake, self).consume_step(feel_tags=tags, consume_tags=[])