

class Body:

    def __init__(self, organs):
        self.organs = organs

    def step(self):
        """perform one step of simulation everywhere - including the brain"""
        for organ in self.organs:
            organ.step()

    def train(self):
        """train on most recently experienced episode"""
        for organ in self.organs:
            organ.train()