from . import utils

class EnvComm(utils.PermanentName):

    def open_connection(self):
        # called separately from __init__
        raise NotImplementedError()

    def close_connection(self):
        # called separately from __init__
        raise NotImplementedError()

    def add_organism_to_env(self, organism):
        raise NotImplementedError()

    def remove_organism_from_env(self, organism):
        raise NotImplementedError()


class UnityEnvComm(EnvComm):

    def __init__(self, env_simulator_ip, env_simulator_port, **kwargs):
        super(UnityEnvComm, self).__init__(**kwargs)

        self.ip = env_simulator_ip
        self.port = env_simulator_port

    def open_connection(self):
        # TODO maybe unnecesary if data is transmitted by GETs and POSTs
        raise NotImplementedError()

    def close_connection(self):
        # TODO maybe unnecesary if data is transmitted by GETs and POSTs
        raise NotImplementedError()

    def add_organism_to_env(self, organism):
        name = organism.get_name
        morphology = {
            node_name: [
                organ.unity3d_primitive
                for organ in organ_sublist
            ] for node_name, organ_sublist
            in organism.organs.items()}
        # TODO: communicate with server
        raise NotImplementedError()

    def remove_organism_from_env(self, organism):
        name = organism.get_name
        # TODO: communicate with server
        raise NotImplementedError()