from .organs import Organ, NodeOrgan, EdgeOrgan
from .organs.energetics import EnergyNode
import utils

import time
import logging
import tensorflow as tf


class Organism(utils.PermanentName):

    def run(self):
        """clocked simulation loop"""
        train_step = 1

        freq = 10
        self.step_num = 1
        self.t_start = time.time()
        self.train_freq = 150 # individual organs can alter the training frequency (eg: brain during sleep)
        while True:
            self.step()
            # wait only if behind schedule
            if self.step_num % self.train_freq == 0:
                train_step += 1
                logging.log(f'train step {train_step}')
                self.train(tf.math.log(train_step / 1000))
                # this may take some time so reset the clocking data afterward
                self.step_num = 0
                self.t_start = time.time()
            t_end = self.t_start + (self.step_num/freq)
            time.sleep(max(0., t_end - time.time()))

    def step(self):
        # TODO make sure energy nodes perform top_down last
        #  so their _free_energy accurately reflects leftovers
        for organ in self.organ_list:
            organ.step()
        self.step_num += 1

    def add_to_env_simulation(self):
        self.env_comm.add_organism_to_env(self)
        # open modality-specific io streams
        for organ in self.organ_list:
            organ.add_to_env_simulation()

    def remove_from_env_simulation(self):
        self.env_comm.remove_organism_from_env(self)
        # close modality-specific io streams
        for organ in self.organ_list:
            organ.remove_from_env_simulation()

    def freeze(self, freeze_to_path):
        # TODO save energy values of self.nodes. No - they are actually organ_graph too
        # Let the individual organ_graph freeze themselves
        for organ in self.organ_list:
            organ.freeze(freeze_to_path)

    def unfreeze(self, unfreeze_from_path):
        # TODO restore energy values of self.nodes. No - they are actually organ_graph too
        # Let the individual organ_graph unfreeze themselves
        for organ in self.organ_list:
            organ.unfreeze(unfreeze_from_path)

    def __init__(self,
                 name,
                 freq,
                 env_comm,
                 organ_graph,
                 d_energy):
        """
        """


        super(Organism, self).__init__(name=name)

        self.freq = freq
        self.env_comm = env_comm
        self.organ_graph = organ_graph
        self.d_energy = d_energy

        self.organ_list = []
        for key, organ_sublist in self.organ_graph.items():
            self.organ_list.extend(organ_sublist)
            if isinstance(key, EnergyNode): # not edges (tuples)
                node = key
                organ_sublist.append(node) # this allows the Node to also connect to its neighbors
                self.organ_list.append(node)
                for node_organ in organ_sublist:
                    node_organ.set_node(node)
                    node_organ.parallel_node_organs = organ_sublist
            elif isinstance(key, tuple) and len(key) == 2:
                src, dst = key
                antiparallel_edge_organs_sublist = self.organ_graph[dst, src] \
                    if (dst, src) in organ_graph.keys() else []
                for edge_organ in organ_sublist:
                    edge_organ.set_nodes(src, dst)
                    edge_organ.parallel_edge_organs = organ_sublist
                    edge_organ.antiparallel_edge_organs = antiparallel_edge_organs_sublist
            else:
                raise TypeError("keys in the organ graph should be NodeOrgan or NodeOrgan 2-tuples")

        for organ in self.organ_list:
            organ.set_organism(self)
