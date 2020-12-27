from .body_plans import Humanoid
from . import env_comm

import logging


if __name__ == '__main__':

    unfreeze_from_path = None # TODO get arg
    freeze_to_path = None # TODO get arg
    env_simulator_ip = None # TODO get arg
    env_simulator_port = None # TODO get arg
    fps = None # TODO get arg

    unity_env_comm = env_comm.UnityEnvComm(
        env_simulator_ip=env_simulator_ip,
        env_simulator_port=env_simulator_port,
        name='environment_simulator')
    unity_env_comm.open_connection()
    humanoid = Humanoid(env_comm=unity_env_comm, fps=fps)
    if unfreeze_from_path:
        logging.log('unfreezing organism from {unfreeze_from_path}')
        humanoid.unfreeze(unfreeze_from_path)
    logging.log('adding organism to environment')
    humanoid.add_to_env_simulation()

    logging.log('starting to run agent')
    humanoid.run()

    logging.log('removing from simulation')
    humanoid.remove_from_env_simulation()

    unity_env_comm.close_connection()

    if freeze_to_path:
        logging.log('freezing organism to {freeze_to_path}')
        humanoid.freeze(freeze_to_path)

    logging.log('exiting normally')