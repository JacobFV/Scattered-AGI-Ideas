from . import utils
import requests

class EnvComm(utils.PermanentName, utils.Stepable):

    def __init__(self, **kwargs):
        self._commands = dict()
        self._responses = dict()

        self._open_connection()

    def __del__(self):
        self._close_connection()

    def __enter__(self): pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()

    def add_command(self, id, cmd):
        self._commands[id] = cmd

    @property
    def get_responses(self):
        return self._responses

    def step(self):
        # subclasses should have already assigned responses to self._responses
        self._commands.clear()

    def _open_connection(self):
        # called by __init__
        raise NotImplementedError()

    def _close_connection(self):
        # called by __del__
        raise NotImplementedError()

    def add_organism_to_env(self, organism):
        raise NotImplementedError()

    def remove_organism_from_env(self, organism):
        raise NotImplementedError()


class GymEnvComm(EnvComm): pass


class TDWEnvComm(EnvComm): pass


class UnityEnvComm(EnvComm):

    def __init__(self, env_simulator_ip, env_simulator_port, **kwargs):
        super(UnityEnvComm, self).__init__(**kwargs)
        self.ip = env_simulator_ip
        self.port = env_simulator_port

    def step(self):
        r = requests.get(f'{self.ip}:{self.port}', params=self._commands)
        self._responses = r.json()
        super(UnityEnvComm, self).step()
        self._commands = dict()

    def open_connection(self):
        # TODO maybe unnecesary if data is transmitted by GETs and POSTs
        raise NotImplementedError()

    def close_connection(self):
        # TODO maybe unnecesary if data is transmitted by GETs and POSTs
        raise NotImplementedError()

    def add_organism_to_env(self, organism):
        morphology = {
            node_name: [
                organ.unity3d_primitive
                for organ in organ_sublist
            ] for node_name, organ_sublist
            in organism.organ_graph.items()}
        self._commands[f'add_organism_{organism.get_name}'] = morphology
        self.step()

    def remove_organism_from_env(self, organism):
        name = organism.get_name
        # TODO: communicate with server
        raise NotImplementedError()