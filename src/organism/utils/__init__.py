import tensorflow as tf
keras = tf.keras
tfkl = keras.layers
K = keras.backend


class Freezable:
    """
    Freezables must first be initialized before unfreezing
    """

    def freeze(self, freeze_to_path):
        raise NotImplementedError()

    def unfreeze(self, unfreeze_from_path):
        raise NotImplementedError()


class SimulationEnvCommunicator:

    def add_to_env_simulation(self):
        raise NotImplementedError()

    def remove_from_env_simulation(self):
        raise NotImplementedError()


class PermanentName:

    def __init__(self, name):
        self._name = name

    @property
    def get_name(self):
        return self._name


def structured_op(dict_obj, op):
    return {k: structured_op(v, op) if isinstance(v, dict) else op(v)
            for k, v in dict_obj.items()}


def pairwise_structured_op(dict_1, dict_2, op):
    return {k1: pairwise_structured_op(v1, v2, op) if isinstance(v1, dict) else op(v1, v2)
            for (k1, v1), (k2, v2) in zip(dict_1.items(), dict_2.items())}


def reduce_sum_dict(dict_obj):
    return reduce_dict(dict_obj, sum)


def reduce_dict(dict_obj, reduce_fn):
    ret_obj = None
    for k, v in dict_obj:
        if isinstance(v, dict):
            ret_obj = reduce_fn(ret_obj, reduce_sum_dict(v))
        else:
            ret_obj = reduce_fn(ret_obj, v)
    return ret_obj
