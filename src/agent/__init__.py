import gene_expression
import organs
import utils
from agent import Agent


import argparse

# CLI:
# new dna_path
# load tar_path
# shared args:
#   env_simulator_ip,
#   env_simulator_port,
#   orchestrator_ip,
#   orchestrator_port

def new_agent_from_dna(dna_path):
    """start new fetus"""
    # TODO load dna as rectangular 2d list of strings
    #       dna = open(dna_path, "r")
    #       dna = json(dna)
    agent = Agent( TODO )
    agent.run()

def load_agent_from_tar(tar_path):
    """resume or clone existing"""
    agent = Agent( TODO )
    # TODO: dir = decompress(tar_path)
    agent.unfreeze(dir)
    agent.run()