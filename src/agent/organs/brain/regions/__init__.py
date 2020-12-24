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