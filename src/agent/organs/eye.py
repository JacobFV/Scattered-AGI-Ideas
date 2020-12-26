"""

    ### NOTE: the eye manages its own data stream

    def add_to_env_simulation(self):
        # TODO open communication channel with self.env_simulator_ip/port
        super(Agent, self).add_to_env_simulation() # open any io streams
        raise NotImplementedError()

    def remove_from_env_simulation(self):
        # TODO close communication channel with self.env_simulator_ip/port
        super(Agent, self).remove_from_env_simulation() # close any io streams
        raise NotImplementedError()
"""