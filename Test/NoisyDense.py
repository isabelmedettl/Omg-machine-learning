from tensorflow.keras import layers
import tensorflow as tf


class NoisyDense(layers.Layer):
    """
    Noisy dense layer to introduce noise to the weights and biases, promoting exploration.
    Now supports activation functions.
    """

    def __init__(self, units, activation=None, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)  # Convert activation argument to activation function

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        # Weight parameters
        self.W_mu = self.add_weight(shape=(self.input_dim, self.units), initializer='random_normal', name='W_mu')
        self.W_sigma = self.add_weight(shape=(self.input_dim, self.units), initializer='random_normal', name='W_sigma')
        self.w_epsilon = tf.random.normal((self.input_dim, self.units))
        # Bias parameters
        self.b_mu = self.add_weight(shape=(self.units,), initializer='random_normal', name='b_mu')
        self.b_sigma = self.add_weight(shape=(self.units,), initializer='random_normal', name='b_sigma')
        self.b_epsilon = tf.random.normal((self.units,))
        super(NoisyDense, self).build(input_shape)  # It's usually not necessary to explicitly call the super().build()

    def call(self, inputs):
        W = self.W_mu + self.W_sigma * self.w_epsilon
        b = self.b_mu + self.b_sigma * self.b_epsilon
        output = tf.matmul(inputs, W) + b
        if self.activation is not None:
            return self.activation(output)  # Apply activation function if not None
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units