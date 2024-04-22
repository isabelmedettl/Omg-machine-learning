from tensorflow.keras import layers
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko
import numpy as np

class NoisyDense(layers.Layer):
    #Noisy dense layer to introduce noise to the weights and biases, promoting exploration.
    #Now supports activation functions.
    def __init__(self, units, activation=None, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)  # Convert activation argument to activation function

    # simpler/classic implementation
    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        # Weight parameters
        self.weight_mu = self.add_weight(shape=(self.input_dim, self.units), initializer='random_normal', name='weight_mu')
        self.weight_sigma = self.add_weight(shape=(self.input_dim, self.units), initializer='random_normal', name='weight_sigma')
        # self.w_epsilon = tf.random.normal((self.input_dim, self.units))
        # Bias parameters
        self.bias_mu = self.add_weight(shape=(self.units,), initializer='random_normal', name='bias_mu')
        self.bias_sigma = self.add_weight(shape=(self.units,), initializer='random_normal', name='bias_sigma')
        # self.b_epsilon = tf.random.normal((self.units,))
        super(NoisyDense, self).build(input_shape)  # It's usually not necessary to explicitly call the super().build()

    # more possiblities for tailoring, but more complex
    """def build(self, input_shape):
        self.input_dim = input_shape[-1]
        mu_range = 1 / np.sqrt(self.input_dim)
        mu_initializer = tf.random_uniform_initializer(-mu_range, mu_range)
        sigma_initializer = tf.constant_initializer(0.5 / np.sqrt(self.units))

        self.weight_mu = tf.Variable(initial_value=mu_initializer(shape=(self.input_dim, self.units), dtype='float32'), trainable=True)
        self.weight_sigma = tf.Variable(initial_value=sigma_initializer(shape=(self.input_dim, self.units), dtype='float32'), trainable=True)
        self.bias_mu = tf.Variable(initial_value=mu_initializer(shape=(self.units,), dtype='float32'), trainable=True)
        self.bias_sigma = tf.Variable(initial_value=sigma_initializer(shape=(self.units,), dtype='float32'), trainable=True)"""

    def call(self, inputs):
        self.weight_epsilon = tf.random.normal((self.input_dim, self.units))
        self.bias_epsilon = tf.random.normal((self.units,))
        w = self.weight_mu + self.weight_sigma * self.weight_epsilon
        b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        output = tf.matmul(inputs, w) + b
        if self.activation is not None:
            return self.activation(output)  # Apply activation function if not None
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units