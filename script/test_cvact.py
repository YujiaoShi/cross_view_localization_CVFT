import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from OriNet_CVACT.input_data_VGG import InputData
from ot_net import *

import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import *
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='TensorFlow implementation.')

parser.add_argument('--network_type',              type=str,   help='network type',      default='CVFT')
args = parser.parse_args()

# --------------  configuration parameters  -------------- #
network_type = args.network_type
data_type = 'CVACT'

batch_size = 32
loss_weight=10

# -------------------------------------------------------- #

def validate(dist_array, top_k):
    accuracy = 0.0
    data_amount = 0.0
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[:, i] < gt_dist)
        if prediction < top_k:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount

    return accuracy


def compute_loss(sat_global, grd_global, utms_x, UTMthres, batch_hard_count=0):

    with tf.name_scope('weighted_soft_margin_triplet_loss'):

        dist_array = 2 - 2 * tf.matmul(sat_global, grd_global, transpose_b=True)
        pos_dist = tf.diag_part(dist_array)

        usefulPair = greater_equal(utms_x[:,:,0], UTMthres, 'good_pair')

        usefulPair = tf.cast(usefulPair, tf.float32)

        if batch_hard_count == 0:

            pair_n = tf.reduce_sum(usefulPair)

            # ground to satellite
            triplet_dist_g2s = (pos_dist - dist_array)*usefulPair

            loss_g2s = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))) / pair_n

            # satellite to ground
            triplet_dist_s2g = (tf.expand_dims(pos_dist, 1) - dist_array)*usefulPair

            loss_s2g = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))) / pair_n

            loss = (loss_g2s + loss_s2g) / 2.0

    return loss



if __name__ == '__main__':
    '''
    Train the network and do the test
    :param start_epoch: the epoch id start to train. The first epoch is 1.
    '''

    # import data
    input_data = InputData()
    # define placeholders

    grd_x = tf.placeholder(tf.float32, [None, 112, 616, 3], name='grd_x')
    sat_x = tf.placeholder(tf.float32, [None, 256, 256, 3], name='sat_x')
    utms_x = tf.placeholder(tf.float32, [None, None, 1], name='utms')

    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)


    # build model
    if network_type == 'CVFT':
        sat_global, grd_global = CVFT(sat_x, grd_x, keep_prob, False)
    elif network_type == 'VGG_conv':
        sat_global, grd_global = VGG_conv(sat_x, grd_x, keep_prob, False)
    elif network_type == 'VGG_gp':
        sat_global, grd_global = VGG_gp(sat_x, grd_x, keep_prob, False)

    out_channel = sat_global.get_shape().as_list()[-1]
    sat_global_descriptor = np.zeros([input_data.get_test_dataset_size(), out_channel])
    grd_global_descriptor = np.zeros([input_data.get_test_dataset_size(), out_channel])

    # set training
    global_step = tf.Variable(0, trainable=False)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print('load model...')
        load_model_path = '../Model/trained_model/CVACT/CVFT/model.ckpt'
        saver.restore(sess, load_model_path)

        # ---------------------- validation ----------------------

        print('validate...')
        print('   compute global descriptors')
        input_data.reset_scan()

        val_i = 0
        while True:
            print('      progress %d' % val_i)
            batch_sat, batch_grd, _ = input_data.next_batch_scan(batch_size)
            if batch_sat is None:
                break
            feed_dict = {sat_x: batch_sat, grd_x: batch_grd, keep_prob: 1.0}
            sat_global_val, grd_global_val = \
                sess.run([sat_global, grd_global], feed_dict=feed_dict)

            sat_global_descriptor[val_i: val_i + sat_global_val.shape[0], :] = sat_global_val
            grd_global_descriptor[val_i: val_i + grd_global_val.shape[0], :] = grd_global_val
            val_i += sat_global_val.shape[0]

        print('   compute accuracy')
        dist_array = 2 - 2 * np.matmul(sat_global_descriptor, np.transpose(grd_global_descriptor))
        top1_percent = int(dist_array.shape[0] * 0.01) + 1
        val_accuracy = np.zeros((1, top1_percent))
        print('start')
        for i in range(top1_percent):
            val_accuracy[0, i] = validate(dist_array, i)

        print(network_type, ' done...')
        print('top1', ':', val_accuracy[0, 1])
        print('top5', ':', val_accuracy[0, 5])
        print('top10', ':', val_accuracy[0, 10])
        print('top1%', ':', val_accuracy[0, -1])

