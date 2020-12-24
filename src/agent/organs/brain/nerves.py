class Sensor:

    def get_observation_space(self):
        raise NotImplementedError()

    def get_observation(self):
        raise NotImplementedError()


class Actuator:

    def get_action_space(self):
        raise NotImplementedError()

    def do_action(self):
        raise NotImplementedError()