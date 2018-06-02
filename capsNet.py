import tensorflow as tf

from config import cfg
from utils import get_batch_data
from utils import softmax
from utils import reduce_sum
from capsLayer import CapsLayer


epsilon = 1e-9


class CapsNet(object):
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.X, self.labels = get_batch_data(cfg.batch_size, cfg.num_threads)
                self.Y = tf.one_hot(self.labels, depth=10, axis=1, dtype=tf.float32)

                self.build_arch()
                self.loss()
                self._summary()

                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer()
                self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)
            else:
                self.X = tf.placeholder(tf.float32, shape=(cfg.batch_size, 28, 28, 1))
                self.labels = tf.placeholder(tf.int32, shape=(cfg.batch_size, ))
                self.Y = tf.reshape(self.labels, shape=(cfg.batch_size, 10, 1))
                self.build_arch()

        tf.logging.info('Seting up the main structure')

    def build_arch(self):
        with tf.variable_scope('Conv1_layer'):
            # simple convolutional layer, first on arch.png
            conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=256,
                                             kernel_size=9, stride=1,
                                             padding='VALID')
            assert conv1.get_shape() == [cfg.batch_size, 20, 20, 256]

        # Primary capsules convolutional layer, second on arch.png
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(num_outputs=32, vec_len=8, layer_type='CONV')
            caps1 = primaryCaps(conv1, kernel_size=9, stride=2)

        # Digit capsules layer, third in arch.png
        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(num_outputs=10, vec_len=16, layer_type='FC')
            self.caps2 = digitCaps(caps1)

        # Decoder structure in Fig. 2
        # 1. Do masking, how:
        with tf.variable_scope('Masking'):
            # a). calc ||v_c||, then do softmax(||v_c||)
            self.v_length = tf.sqrt(reduce_sum(tf.square(self.caps2),
                                               axis=2, keepdims=True) + epsilon)
            self.softmax_v = softmax(self.v_length, axis=1)

            # b). pick out the index of max softmax val of the 10 caps
            self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
            self.argmax_idx = tf.reshape(self.argmax_idx, shape=(cfg.batch_size, ))

            self.masked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)))
            self.v_length = tf.sqrt(reduce_sum(tf.square(self.caps2), axis=2, keepdims=True) + epsilon)

        # 2. Reconstructe the MNIST images with 3 FC layers, reconstruct.png
        with tf.variable_scope('Decoder'):
            vector_j = tf.reshape(self.masked_v, shape=(cfg.batch_size, -1))
            fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
            self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)

    def loss(self):
        # 1. The margin loss, margin-loss.png

        # [batch_size, 10, 1, 1]
        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., cfg.m_plus - self.v_length))
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., self.v_length - cfg.m_minus))
        assert max_l.get_shape() == [cfg.batch_size, 10, 1, 1]

        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))

        T_c = self.Y
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # 2. The reconstruction loss
        origin = tf.reshape(self.X, shape=(cfg.batch_size, -1))
        squared = tf.square(self.decoded - origin)
        self.reconstruction_err = tf.reduce_mean(squared)

        # 3. Total loss
        self.total_loss = self.margin_loss + cfg.regularization_scale * self.reconstruction_err

    # Summary
    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/margin_loss', self.margin_loss))
        train_summary.append(tf.summary.scalar('train/reconstruction_loss', self.reconstruction_err))
        train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))
        recon_img = tf.reshape(self.decoded, shape=(cfg.batch_size, 28, 28, 1))
        train_summary.append(tf.summary.image('reconstruction_img', recon_img))
        self.train_summary = tf.summary.merge(train_summary)

        correct_prediction = tf.equal(tf.to_int32(self.labels), self.argmax_idx)
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
