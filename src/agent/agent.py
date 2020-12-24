from gene_expression import gene_loader
from body import Body


class Agent:

    def __init__(self,
                 name,
                 dna,
                 simulator_addr=None,
                 orchestrator_addr=None):
        self._name = name
        self.simulator_addr = simulator_addr
        self.orchestrator_addr = orchestrator_addr

        self.dna = dna
        self.rna = None
        self.gene_expression_fns = [gene_loader]

        self.body = Agent.Body()

        if self.orchestrator_addr is not None:
            self.reverb_addr =  # query reverb addr
        else:
            # make new reverb server
            self.reverb_addr = "TODO"
            raise NotImplementedError()

    def step(self):
        for fn in self.gene_expression_fns[:]:
            fn(self)
        for organ in self.body.get_all_organs():
            organ.step()

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

    def train(self):
        for organ in self.body.get_all_organs():
            organ.train()

    def save_frozen_copy(self):
        """saves to tar"""
        # make parent dir
        # save self data
        # save each sub-organ
        # compress
        raise NotImplementedError()

    @staticmethod
    def init_from_freeze(dir):
        pass

    @property
    def get_name(self):
        return self._name

    def add_to_simulation_env(self):
        raise NotImplementedError()

    def remove_from_simulation_env(self):
        raise NotImplementedError()
