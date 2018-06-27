from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflib as tl

conv = partial(slim.conv2d, activation_fn=None)
dconv = partial(slim.conv2d_transpose, activation_fn=None)
fc = partial(tl.flatten_fully_connected, activation_fn=None)
relu = tf.nn.relu
lrelu = tf.nn.leaky_relu
batch_norm = partial(slim.batch_norm, scale=True, updates_collections=None)


def mlp_mnist():

    def Enc(img, z_dim, dim=512, is_training=True):
        fc_relu = partial(fc, activation_fn=relu)

        with tf.variable_scope('Enc', reuse=tf.AUTO_REUSE):
            y = fc_relu(img, dim)
            y = fc_relu(y, dim * 2)
            z_mu = fc(y, z_dim)
            z_log_sigma_sq = fc(y, z_dim)
            return z_mu, z_log_sigma_sq

    def Dec(z, dim=512, channels=1, is_training=True):
        fc_relu = partial(fc, activation_fn=relu)

        with tf.variable_scope('Dec', reuse=tf.AUTO_REUSE):
            y = fc_relu(z, dim * 2)
            y = fc_relu(y, dim)
            y = tf.tanh(fc(y, 28 * 28 * channels))
            img = tf.reshape(y, [-1, 28, 28, channels])
            return img

    return Enc, Dec


def conv_mnist():

    def Enc(img, z_dim, dim=64, is_training=True):
        bn = partial(batch_norm, is_training=is_training)
        conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu)

        with tf.variable_scope('Enc', reuse=tf.AUTO_REUSE):
            y = conv_bn_lrelu(img, dim, 5, 2)
            y = conv_bn_lrelu(y, dim * 2, 5, 2)
            z_mu = fc(y, z_dim)
            z_log_sigma_sq = fc(y, z_dim)
            return z_mu, z_log_sigma_sq

    def Dec(z, dim=64, channels=1, is_training=True):
        bn = partial(batch_norm, is_training=is_training)
        dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu)

        with tf.variable_scope('Dec', reuse=tf.AUTO_REUSE):
            y = relu(fc(z, 7 * 7 * dim * 2))
            y = tf.reshape(y, [-1, 7, 7, dim * 2])
            y = dconv_bn_relu(y, dim * 1, 5, 2)
            img = tf.tanh(dconv(y, channels, 5, 2))
            return img

    return Enc, Dec


def conv_64():

    def Enc(img, z_dim, dim=64, is_training=True):
        bn = partial(batch_norm, is_training=is_training)
        conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu)

        with tf.variable_scope('Enc', reuse=tf.AUTO_REUSE):
            y = conv_bn_lrelu(img, dim, 5, 2)
            y = conv_bn_lrelu(y, dim * 2, 5, 2)
            y = conv_bn_lrelu(y, dim * 4, 5, 2)
            y = conv_bn_lrelu(y, dim * 8, 5, 2)
            z_mu = fc(y, z_dim)
            z_log_sigma_sq = fc(y, z_dim)
            return z_mu, z_log_sigma_sq

    def Dec(z, dim=64, channels=3, is_training=True):
        bn = partial(batch_norm, is_training=is_training)
        dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu)

        with tf.variable_scope('Dec', reuse=tf.AUTO_REUSE):
            y = relu(fc(z, 4 * 4 * dim * 8))
            y = tf.reshape(y, [-1, 4, 4, dim * 8])
            y = dconv_bn_relu(y, dim * 4, 5, 2)
            y = dconv_bn_relu(y, dim * 2, 5, 2)
            y = dconv_bn_relu(y, dim * 1, 5, 2)
            img = tf.tanh(dconv(y, channels, 5, 2))
            return img

    return Enc, Dec
