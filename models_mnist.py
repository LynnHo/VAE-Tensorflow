from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim

from functools import partial

conv = partial(slim.conv2d, activation_fn=None)
dconv = partial(slim.conv2d_transpose, activation_fn=None)
fc = partial(ops.flatten_fully_connected, activation_fn=None)
relu = tf.nn.relu
lrelu = partial(ops.leak_relu, leak=0.2)
batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, updates_collections=None)


def encoder(img, z_dim, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope('encoder', reuse=reuse):
        y = conv_bn_lrelu(img, dim, 5, 2)
        y = conv_bn_lrelu(y, dim * 2, 5, 2)
        z_mu = fc(y, z_dim)
        z_log_sigma_sq = fc(y, z_dim)
        return z_mu, z_log_sigma_sq


def decoder(z, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

    with tf.variable_scope('decoder', reuse=reuse):
        y = fc(z, 7 * 7 * dim * 2)
        y = tf.reshape(y, [-1, 7, 7, dim * 2])
        y = relu(bn(y))
        y = dconv_bn_relu(y, dim * 1, 5, 2)
        img = tf.tanh(dconv(y, 1, 5, 2))
        return img
