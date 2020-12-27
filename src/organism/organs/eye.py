from . import NodeOrgan

import numpy as np
import gym
import socket
import logging


class Eye(NodeOrgan):

    def __init__(self, height, width, channels=3, **kwargs):

        super(Eye, self).__init__(**kwargs)

        self.width = width
        self.height = height
        self.channels = channels

        self.image = np.zeros((height, width, channels))

    def get_observation_space(self):
        return gym.spaces.Box(low=0., high=1., shape=(self.height, self.width, self.channels))

    def get_action_space(self):
        # [pitch, yaw, dof, focal_length]
        return gym.spaces.Box(low=0., high=1., shape=(4,))

class UnityEye(Eye):

    def __init__(self, **kwargs):
        super(UnityEye, self).__init__(**kwargs)

    def step(self):
        # update self.image
        raw = self.conn.recv()

        # perform action
        # rotate eye
        # adjust pitch and yaw

    def add_to_env_simulation(self):
        # by now, the unity engine has already made a camera object
        # we just need to query for and connect to it

        unity_camera_name = f'{self.agent.get_name}_{self.get_name}'

        ip = self.agent.env_comm.env_simulator_ip

        # TODO query ip : self.agent.env_comm.env_simulator_port for
        # the port that unity_camera_name belongs to
        port = None

        self.camera_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.camera_socket.bind((ip, port))
        self.camera_socket.listen()
        self.conn, addr = self.camera_socket.accept()
        if not self.conn:
            logging.log(f"eye {self.get_name} could not connect to {ip}:{port} {addr}")

    def remove_from_env_simulation(self):
        # TODO close communication channel with self.env_simulator_ip/port
        raise NotImplementedError()