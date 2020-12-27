from . import NodeOrgan

class Intake(NodeOrgan):

    def __init__(self, consumable_tag="food", embedding_network=None, output_dim):

        if not embedding_network:
            embedding_network = tfkl . #TODO

    def step(self):
        # actually

    def get_observation_space(self):
        return gym.spaces.Box(low=0., high=1., shape=(1,))

    def get_action_space(self):
        # [pitch, yaw, dof, focal_length]
        return gym.spaces.Box(low=0., high=1., shape=(4,))