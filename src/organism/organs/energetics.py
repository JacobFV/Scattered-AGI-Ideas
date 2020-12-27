converter
transfer
storage


def step(self):
    raise NotImplementedError()


def train(self):
    raise NotImplementedError()


def save_frozen_copy(self):
    raise NotImplementedError()


def restore_from_freeze(self, dir):
    pass