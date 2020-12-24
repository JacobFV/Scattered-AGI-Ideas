from ... import utils
from .. import Organ


class Brain(Organ,
            utils.PermanentName):

    def __init__(self, **kwargs):
        """
        params:
        name
        agent
        """

        self.pfc = TODO
        self.V124 = TODO
        self.brocas = TODO
        TODO

        regions = [self.pfc, self.V124, self.brocas]
        kwargs['freezables'] = regions
        kwargs['trainables'] = regions
        kwargs['stepables'] = regions

        super(Brain, self).__init__(**kwargs)

    def step(self):
        # pre-ops
        super(Brain, self).step()
        # post-ops

    def train(self):
        # pre-ops
        super(Brain, self).train()
        # post-ops

    def freeze(self):
        # pre-ops
        super(Brain, self).freeze(dir)
        # post-ops

    def unfreeze(self):
        # pre-ops
        super(Brain, self).unfreeze(dir)
        # post-ops
