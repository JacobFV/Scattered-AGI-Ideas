from .... import utils


class BrainRegion(utils.Freezable,
                  utils.Stepable,
                  utils.Trainable,
                  utils.PermanentName):

    def step(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def freeze(self, dir):
        raise NotImplementedError()

    def unfreeze(self, dir):
        raise NotImplementedError()


class PeripheralBrainRegion(BrainRegion): pass


class GlobalWorkspaceRegion(BrainRegion):

    @property
    def get_latent(self):
        raise NotImplementedError("this method should be overriden in subclasses")