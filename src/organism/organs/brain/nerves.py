class Sensor:

    def get_observation_space(self):
        raise NotImplementedError()

    def get_observation(self):
        return self.observation


class Actuator:

    def get_action_space(self):
        raise NotImplementedError()

    def set_action(self, action):
        self.action = action
