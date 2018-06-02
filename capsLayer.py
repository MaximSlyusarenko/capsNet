import numpy as np
import tensorflow as tf

from config import cfg
from utils import reduce_sum
from utils import softmax


epsilon = 1e-9


class CapsLayer(object):
    def __init__(self, num_outputs, vec_len, with_routing=True, layer_type='FC'):
        self.num_outputs = num_outputs
        self.vec_len = vec_len # output vector length
        self.layer_type = layer_type

    def __call__(self, input, kernel_size=None, stride=None):
        # Convolutional
        if self.layer_type == 'CONV': # convolutional layer, no routing
            self.kernel_size = kernel_size
            self.stride = stride

            capsules = []
            for i in range(self.vec_len):
                with tf.variable_scope('ConvUnit_' + str(i)):
                    caps_i = tf.contrib.layers.conv2d(input, self.num_outputs,
                                                        self.kernel_size, self.stride,
                                                        padding="VALID", activation_fn=None)
                    caps_i = tf.reshape(caps_i, shape=(cfg.batch_size, -1, 1, 1))
                    capsules.append(caps_i)
            capsules = tf.concat(capsules, axis=2)

            capsules = squash(capsules)
            return(capsules)

        # fully-connected
        if self.layer_type == 'FC': # fully-connected layer, with routing
            # Reshape the input into [batch_size, 1152, 1, 8, 1]
            self.input = tf.reshape(input, shape=(cfg.batch_size, -1, 1, input.shape[-2].value, 1))

            with tf.variable_scope('routing'):
                # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
                b_IJ = tf.constant(np.zeros([cfg.batch_size, input.shape[1].value, self.num_outputs, 1, 1], dtype=np.float32))
                capsules = routing(self.input, b_IJ)
                capsules = tf.squeeze(capsules, axis=1)

            return(capsules)


def routing(input, b_IJ):
    # W - start of routing algorithm, initialization
    W = tf.get_variable('Weight', shape=(1, 1152, 160, 8, 1), dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=cfg.stddev))
    biases = tf.get_variable('bias', shape=(1, 1, 10, 16, 1))

    # u_hat.png
    input = tf.tile(input, [1, 1, 160, 1, 1])

    u_hat = reduce_sum(W * input, axis=3, keepdims=True)
    u_hat = tf.reshape(u_hat, shape=[-1, 1152, 10, 16, 1])

    # In forward, u_hat_no_back_propogation = u_hat; in backward, no gradient passed back from u_hat_no_back_propogation to u_hat
    u_hat_no_back_propogation = tf.stop_gradient(u_hat, name='stop_gradient')

    # routing.png, cycle
    for r_iter in range(cfg.iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            c_IJ = softmax(b_IJ, axis=2)

            # last iteration, use u_hat for back-propogation
            if r_iter == cfg.iter_routing - 1:
                s_J = tf.multiply(c_IJ, u_hat)
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(s_J)
            elif r_iter < cfg.iter_routing - 1:  # Inner routing iterations, no back-propogation, so use u_hat_no_back_propogation
                s_J = tf.multiply(c_IJ, u_hat_no_back_propogation)
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(s_J)
                v_J_tiled = tf.tile(v_J, [1, 1152, 1, 1, 1])
                u_produce_v = reduce_sum(u_hat_no_back_propogation * v_J_tiled, axis=3, keepdims=True)
                b_IJ += u_produce_v

    return(v_J)


# squashing.png
def squash(vector):
    vec_squared_norm = reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector
    return(vec_squashed)
