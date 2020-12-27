from .organs import Organ, NodeOrgan, EdgeOrgan
import utils

import time


class Organism(utils.Freezable,
               utils.Stepable,
               utils.Trainable,
               utils.SimulationEnvCommunicator,
               utils.PermanentName):

    def run(self):
        """clocked simulation loop"""
        freq = 10
        self.step_num = 0
        self.t_start = time.time()
        while True:
            self.step()
            # wait only if behind schedule
            t_end = self.t_start + (self.step_num/freq)
            time.sleep(max(0., t_end - time.time()))

    def train(self):
        super(Organism, self).train()
        # this may take some time so reset the clocking data afterward
        self.step_num = 0
        self.t_start = time.time()

    def add_to_env_simulation(self):
        self.env_comm.add_organism_to_env(self)
        super(Organism, self).add_to_env_simulation() # open modality-specific io streams

    def remove_from_env_simulation(self):
        self.env_comm.remove_organism_from_env(self)
        super(Organism, self).remove_from_env_simulation() # close modality-specific io streams

    def freeze(self, freeze_to_path):
        # no custom logic here. Let the individual organs freeze themselves
        super(Organism, self).freeze(freeze_to_path)

    def unfreeze(self, unfreeze_from_path):
        # no custom logic here. Let the individual organs unfreeze themselves
        super(Organism, self).unfreeze(unfreeze_from_path)

    def __init__(self,
                 name,
                 organs,
                 env_comm):
        """
        name
        organs: dict of str:NodeOrgan or (str,str):EdgeOrgan
        env_simulator_ip
        env_simulator_port
        """

        self.all_organs_list = []

        super(Organism, self).__init__(
            name=name,
            freezables=self.all_organs_list,
            stepables=self.all_organs_list,
            simulation_env_communicators=self.all_organs_list,
            trainables=self.all_organs_list)

        self.organs = organs
        self.env_comm = env_comm
        self.coordinating_nodes = dict()
        for name, organ_sublist in organs.items():
            self.all_organs_list.extend(organ_sublist)
            if isinstance(name, str): # only nodes
                self.coordinating_nodes[name]=CoordinatingNode(name=name)
        for node_name, node in self.coordinating_nodes:
            for name, organ_sublist in organs.items():
                if isinstance(name, str):
                    if node_name == name:
                        node.node_organs = organ_sublist
                        for organ in organ_sublist:
                            organ._set_node(node)
                elif isinstance(name, tuple):
                    src, dst = name
                    if node_name == src:
                        src_node = node
                        dst_node = self.coordinating_nodes[dst]
                        src_node.outgoing_edge_organs = organ_sublist
                        for organ in organ_sublist:
                            organ._set_nodes(src, dst)
                    elif node_name == dst:
                        src_node = self.coordinating_nodes[src]
                        dst_node = node
                        src_node.incoming_edge_organs = organ_sublist
                        for organ in organ_sublist:
                            organ._set_nodes(src, dst)
                    else: pass
                else:
                    raise TypeError("keys in the organ graph should be str or str tuples")


class CoordinatingNode(utils.PermanentName):
    def __init__(self, **kwargs):
        super(CoordinatingNode, self).__init__(**kwargs)
        self.node_organs = []
        self.incoming_edge_organs = []
        self.outgoing_edge_organs = []
