from organs.brain import Brain
import utils

import time


class Agent(utils.Freezable,
            utils.Stepable,
            utils.Trainable,
            utils.SimulationEnvCommunicator,
            utils.PermanentName):

    def run(self):
        """clocked simulation loop"""
        freq = 60
        self.step_num = 0
        self.t_start = time.time()
        while True:
            self.step()
            # wait only if behind schedule
            t_end = self.t_start + (self.step_num/freq)
            time.sleep(max(0, t_end - time.time()))

    def sleep(self):
        self.train()
        # TODO: continue to simulate body while training brain
        # TODO: communicate with orchestrator during sleep
        # TODO:     orchestrator may give larger compute architecture
        # TODO:     or save and shut down the agent
        self.step_num = 0
        self.t_start = time.time()
        raise NotImplementedError()

    def add_to_env_simulation(self):
        # TODO open communication channel with self.env_simulator_ip/port
        super(Agent, self).add_to_env_simulation() # open any io streams
        raise NotImplementedError()

    def remove_from_env_simulation(self):
        # TODO close communication channel with self.env_simulator_ip/port
        super(Agent, self).remove_from_env_simulation() # close any io streams
        raise NotImplementedError()

    def freeze(self, dir):
        # TODO save structure of all organs
        raise NotImplementedError()

    def unfreeze(self, dir):
        # TODO re-create all organs as node or edge organs and unfreeze each organ
        raise NotImplementedError()

    @property
    def get_all_organs(self):
        """ walks along graph from brain to all connected """
        def list_graph(node, bucket):
            for edge in node.outgoing_edges:
                if edge not in bucket:
                    bucket.append(edge)
                    for node in edge.dst_nodes:
                        if node not in bucket:
                            bucket.append(node)
                            list_graph(node, bucket)

            for edge in node.incoming_edges:
                if edge not in bucket:
                    bucket.append(edge)
                    for node in edge.src_nodes:
                        if node not in bucket:
                            bucket.append(node)
                            list_graph(node, bucket)

            return bucket

        return list_graph(self.brain, bucket=[])

    def __init__(self,
                 name,
                 dna,
                 env_simulator_ip,
                 env_simulator_port,
                 orchestrator_ip,
                 orchestrator_port):
        """
        name
        dna
        env_simulator_addr
        orchestrator_addr
        reverb_addr
        """

        super(Agent, self).__init__(
            name=name,
            freezables=self.get_all_organs,
            stepables=self.get_all_organs,
            simulation_env_communicators=self.get_all_organs,
            trainables=self.get_all_organs)

        self.env_simulator_ip = env_simulator_ip
        self.env_simulator_port = env_simulator_port
        self.orchestrator_ip = orchestrator_ip
        self.orchestrator_port = orchestrator_port

        self.brain = Brain(agent=self)