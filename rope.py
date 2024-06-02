import tensorflow as tf
import numpy as np
class RoPE(tf.keras.layers.Layer):
  def __init__(self, d_model, max_position):
    super(RoPE, self).__init__()
    self.d_model = d_model
    self.max_position = max_position
    assert d_model % 2 == 0


  def call(self, inputs):
    positions = tf.range(0, self.max_position, dtype=tf.float32)
    theta = 10000.0 ** (-2 * positions / self.d_model)
    seq_length = tf.shape(inputs)[1]

    sin_coeff = np.zeros_like(inputs)

    sin_coeff[1::2] = inputs[1::2]
    sin_coeff[0::2] = -inputs[0::2]

    sin_matrix = np.zeros((self.d_model, self.d_model))
    cos_matrix = np.zeros((self.d_model, self.d_model))

    sin_matrix[:, 0::2] = np.sin(theta * positions)[0]
    sin_matrix[:, 1::2] = np.sin(theta * positions)[0]

    cos_matrix[0::2] = np.cos(theta * positions)[0]
    cos_matrix[1::2] = np.cos(theta * positions)[0]

    output = tf.matmul(inputs, cos_matrix) + tf.matmul(sin_coeff, sin_matrix)
    return tf.cast(output, tf.float32)