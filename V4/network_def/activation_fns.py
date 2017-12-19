import tensorflow as tf


def prelu(_x):
    alpha = tf.get_variable("alpha", shape=_x.get_shape()[-1],
                            dtype=_x.dtype,
                            initializer=tf.constant_initializer(0.0))
    return tf.maximum(0.0, _x) + alpha * tf.minimum(0.0, _x)
