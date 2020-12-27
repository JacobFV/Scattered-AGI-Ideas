import tensorflow as tf


def exponential_periods(fundamental_period, additional_overtones=0, base=tf.exp(1)):
    """makes exponentially slower periods

    base: e (default), 2, 3, -1.618, any number"""
    return fundamental_period * (base ** tf.range(additional_overtones + 1))
