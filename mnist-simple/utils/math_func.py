import math
import numpy as np
import tensorflow as tf


# log10 for tf.variable
def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

# log10 for numpy.ndarray
def np_log10(x):
    numerator = np.log(x)
    denominator = np.log(10)
    return numerator / denominator

# log10
def math_log10(x):
    return math.log10(x)





