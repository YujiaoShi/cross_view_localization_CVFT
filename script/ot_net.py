import tensorflow as tf
import numpy as np

from VGG import VGG16
from split_feature import split_feature


def sinkhorn(log_alpha, n_iters=20):
    """Performs incomplete Sinkhorn normalization to log_alpha.
    By a theorem by Sinkhorn and Knopp [1], a sufficiently well-behaved matrix
    with positive entries can be turned into a doubly-stochastic matrix
    (i.e. its rows and columns add up to one) via the succesive row and column
    normalization.
    -To ensure positivity, the effective input to sinkhorn has to be
    exp(log_alpha) (elementwise).
    -However, for stability, sinkhorn works in the log-space. It is only at
    return time that entries are exponentiated.
    [1] Sinkhorn, Richard and Knopp, Paul.
    Concerning nonnegative matrices and doubly stochastic
    matrices. Pacific Journal of Mathematics, 1967
    Args:
    log_alpha: 2D tensor (a matrix of shape [N, N])
    or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
    n_iters: number of sinkhorn iterations (in practice, as little as 20
    iterations are needed to achieve decent convergence for N~100)
    Returns:
    A 3D tensor of close-to-doubly-stochastic matrices (2D tensors are
    converted to 3D tensors with batch_size equals to 1)
    """
    n = log_alpha.get_shape().as_list()[1]
    log_alpha = tf.reshape(log_alpha, [-1, n, n])

    for _ in range(n_iters):
        log_alpha -= tf.reshape(tf.reduce_logsumexp(log_alpha, axis=2), [-1, n, 1])
        log_alpha -= tf.reshape(tf.reduce_logsumexp(log_alpha, axis=1), [-1, 1, n])
    return tf.exp(log_alpha)


def CVFT(x_sat, x_grd, keep_prob, trainable):
    def conv_layer(x, kernel_dim, input_dim, output_dim, stride, trainable, activated,
                   name='ot_conv', activation_function=tf.nn.relu):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):  # reuse=tf.AUTO_REUSE
            weight = tf.get_variable(name='weights', shape=[kernel_dim, kernel_dim, input_dim, output_dim],
                                     trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable(name='biases', shape=[output_dim],
                                   trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())

            out = tf.nn.conv2d(x, weight, strides=[1, stride, stride, 1], padding='SAME') + bias

            if activated:
                out = activation_function(out)

            return out

    def fc_layer(x, trainable, name='ot_fc'):
        height, width, channel = x.get_shape().as_list()[1:]
        assert channel == 1
        in_dimension = height * width
        out_dimension = in_dimension ** 2

        input_feature = tf.reshape(x, [-1, height * width])

        with tf.variable_scope(name):
            weight = tf.get_variable(name='weights', shape=[in_dimension, out_dimension],
                                     trainable=trainable,
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.005),
                                     regularizer=tf.contrib.layers.l2_regularizer(0.01))
            bias = tf.get_variable(name='biases', shape=[out_dimension],
                                   trainable=trainable,
                                   initializer=tf.constant_initializer(np.eye(in_dimension).reshape(in_dimension ** 2)))
            out = tf.matmul(input_feature, weight) + bias

            out = tf.reshape(out, [-1, in_dimension, in_dimension])

        return out

    def ot(input_feature, trainable, name='ot'):
        height, width, channel = input_feature.get_shape().as_list()[1:]
        conv_feature = conv_layer(input_feature, kernel_dim=1, input_dim=channel, output_dim=1, stride=1,
                                  trainable=trainable, activated=True, name=name + 'ot_conv')
        fc_feature = fc_layer(conv_feature, trainable, name=name + 'ot_fc')
        ot_matrix = sinkhorn(fc_feature * (-100.))

        return ot_matrix

    def apply_ot(input_feature, ot_matrix):

        height, width, channel = input_feature.get_shape().as_list()[1:]
        in_dimension = ot_matrix.get_shape().as_list()[1]

        reshape_input = tf.transpose(tf.reshape(input_feature, [-1, in_dimension, channel]), [0, 2, 1])
        # shape = [batch, channel, in_dimension]

        out = tf.einsum('bci, bio -> bco', reshape_input, ot_matrix)
        output_feature = tf.reshape(tf.transpose(out, [0, 2, 1]), [-1, height, width, channel])

        return output_feature

    ############## VGG module #################

    vgg_grd = VGG16()
    grd_vgg = vgg_grd.VGG16_conv(x_grd, keep_prob, trainable, 'VGG_grd')
    grd_vgg = conv_layer(grd_vgg, kernel_dim=3, input_dim=512, output_dim=64, stride=2, trainable=trainable,
                         activated=True, name='grd_conv')

    vgg_sat = VGG16()
    sat_vgg = vgg_sat.VGG16_conv(x_sat, keep_prob, trainable, 'VGG_sat')
    sat_vgg = conv_layer(sat_vgg, kernel_dim=3, input_dim=512, output_dim=64, stride=2, trainable=trainable,
                         activated=True, name='sat_conv')

    ############## resize #################
    height, width, channel = sat_vgg.get_shape().as_list()[1:]

    grd_vgg = tf.image.resize_bilinear(grd_vgg, [height, width])

    ############## OT module ######################

    ot_matrix_grd_branch = ot(grd_vgg, trainable, name='ot_grd_branch')
    grd_ot = apply_ot(grd_vgg, ot_matrix_grd_branch)

    sat_ot = sat_vgg

    ################# reshape ###################

    grd_height, grd_width, grd_channel = grd_ot.get_shape().as_list()[1:]
    grd_global = tf.reshape(grd_ot, [-1, grd_height * grd_width * grd_channel])

    sat_height, sat_width, sat_channel = sat_ot.get_shape().as_list()[1:]
    sat_global = tf.reshape(sat_ot, [-1, sat_height * sat_width * sat_channel])

    return tf.nn.l2_normalize(sat_global, dim=1), tf.nn.l2_normalize(grd_global, dim=1)


def VGG_conv(x_sat, x_grd, keep_prob, trainable):
    def conv_layer(x, kernel_dim, input_dim, output_dim, stride, trainable, activated,
                   name='ot_conv', activation_function=tf.nn.relu):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):  # reuse=tf.AUTO_REUSE
            weight = tf.get_variable(name='weights', shape=[kernel_dim, kernel_dim, input_dim, output_dim],
                                     trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable(name='biases', shape=[output_dim],
                                   trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())

            out = tf.nn.conv2d(x, weight, strides=[1, stride, stride, 1], padding='SAME') + bias

            if activated:
                out = activation_function(out)

            return out

    ############## VGG module #################

    vgg_grd = VGG16()
    grd_vgg = vgg_grd.VGG16_conv(x_grd, keep_prob, trainable, 'VGG_grd')
    grd_vgg = conv_layer(grd_vgg, kernel_dim=3, input_dim=512, output_dim=64, stride=2, trainable=trainable,
                         activated=True, name='grd_conv')

    vgg_sat = VGG16()
    sat_vgg = vgg_sat.VGG16_conv(x_sat, keep_prob, trainable, 'VGG_sat')
    sat_vgg = conv_layer(sat_vgg, kernel_dim=3, input_dim=512, output_dim=64, stride=2, trainable=trainable,
                             activated=True, name='sat_conv')

    ############## resize #################
    height, width, channel = sat_vgg.get_shape().as_list()[1:]

    grd_vgg = tf.image.resize_bilinear(grd_vgg, [height, width])

    ############## reshape #################
    grd_height, grd_width, grd_channel = grd_vgg.get_shape().as_list()[1:]
    grd_global = tf.reshape(grd_vgg, [-1, grd_height * grd_width * grd_channel])

    sat_height, sat_width, sat_channel = sat_vgg.get_shape().as_list()[1:]
    sat_global = tf.reshape(sat_vgg, [-1, sat_height * sat_width * sat_channel])

    return tf.nn.l2_normalize(sat_global, dim=1), tf.nn.l2_normalize(grd_global, dim=1)


def VGG_gp(x_sat, x_grd, keep_prob, trainable):

    ############## VGG module #################
    vgg_grd = VGG16()
    grd_vgg = vgg_grd.VGG16_conv(x_grd, keep_prob, trainable, 'VGG_grd')

    vgg_sat = VGG16()
    sat_vgg = vgg_sat.VGG16_conv(x_sat, keep_prob, trainable, 'VGG_sat')

    ############## Global pooling #################
    grd_height, grd_width, grd_channel = grd_vgg.get_shape().as_list()[1:]
    grd_global = tf.nn.max_pool(grd_vgg, [1, grd_height, grd_width, 1], [1, 1, 1, 1], padding='VALID')
    grd_global = tf.reshape(grd_global, [-1, grd_channel])

    sat_height, sat_width, sat_channel = sat_vgg.get_shape().as_list()[1:]
    sat_global = tf.nn.max_pool(sat_vgg, [1, sat_height, sat_width, 1], [1, 1, 1, 1], padding='VALID')
    sat_global = tf.reshape(sat_global, [-1, sat_channel])

    return tf.nn.l2_normalize(sat_global, dim=1), tf.nn.l2_normalize(grd_global, dim=1)


