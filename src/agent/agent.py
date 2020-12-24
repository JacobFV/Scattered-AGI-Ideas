from gene_expression import GeneLoader
from body import Body
import utils


class Agent(utils.Freezable,
            utils.Stepable,
            utils.Trainable,
            utils.PermanentName):


    def __init__(self,
                 name,
                 dna,
                 env_simulator_addr,
                 orchestrator_addr,
                 reverb_addr):
        """
        name
        dna
        env_simulator_addr
        orchestrator_addr
        reverb_addr
        """

        self.env_simulator_addr = env_simulator_addr
        self.orchestrator_addr = orchestrator_addr
        self.reverb_addr = reverb_addr

        self.dna = dna
        self.rna = None
        self.gene_expression_fns = [GeneLoader()]

        self.body = Body()

        all_organs = self.body.get_all_organs
        super(Agent, self).__init__(
            name=name,
            freezables=all_organs,
            stepables=all_organs + self.gene_expression_fns,
            trainables=all_organs)

    def sleep(self):
        self.train()
        # TODO: continue to simulate body while training brain
        raise NotImplementedError()

    def hibernate(self):
        # TODO: first, sleep a lot
        # start_request_compute_refit(self)
        # if should_I_send_frozen_copy(name) == True:
        #    self.save_frozen_copy()
        #    finish_request_compute_refit(name, frozen_agent: tar_file) -> void
        # else:
        #    exit()
        raise NotImplementedError()

    def add_to_env_simulation(self):
        raise NotImplementedError()

    def remove_from_env_simulation(self):
        raise NotImplementedError()
