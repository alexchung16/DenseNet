#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File DenseNet121_slim.py
# @ Description :
# @ Author alexchung
# @ Time 25/11/2019 PM 17:51

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops


class DenseNet121():
    """
    Inception v1
    """
    def __init__(self, input_shape, num_classes, batch_size, num_samples_per_epoch, num_epoch_per_decay,
                 decay_rate, learning_rate, keep_prob=0.8, weight_decay=1e-4,reduction=0.5, grow_rate=32,
                     num_filters=64, num_dense_block=4, num_layers=None, batch_norm_decay=0.9997,
                 batch_norm_epsilon=1.1e-5, batch_norm_scale=False, batch_norm_fused=True, is_pretrain=False,
                 reuse=tf.AUTO_REUSE):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.decay_steps = int(num_samples_per_epoch / batch_size * num_epoch_per_decay)
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.num_filters = num_filters
        self.num_dense_block = num_dense_block
        self.num_layers = num_layers
        self.reduction = reduction
        self.grow_rate = grow_rate
        self.batch_norm_decay = batch_norm_decay
        self.batch_norm_epsilon = batch_norm_epsilon
        self.batch_norm_scale = batch_norm_scale
        self.batch_norm_fused = batch_norm_fused
        self.is_pretrain = is_pretrain
        self.reuse = reuse
        # self.initializer = tf.random_normal_initializer(stddev=0.1)
        # add placeholder (X,label)
        self.raw_input_data = tf.compat.v1.placeholder (tf.float32, shape=[None, input_shape[0], input_shape[1], input_shape[2]],
                                                        name="input_images")
        # y [None,num_classes]
        self.raw_input_label = tf.compat.v1.placeholder (tf.float32, shape=[None, self.num_classes], name="class_label")
        self.is_training = tf.compat.v1.placeholder_with_default(input=False, shape=(), name='is_training')

        self.global_step = tf.train.get_or_create_global_step()
        self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")

        # logits
        self.logits =  self.inference(self.raw_input_data, scope='densenet121')
        # # computer loss value
        self.loss = self.losses(labels=self.raw_input_label, logits=self.logits, scope='Loss')
        # train operation
        self.train = self.training(self.learning_rate, self.global_step, loss=self.loss)
        self.accuracy = self.evaluate_batch(self.logits, self.raw_input_label) / batch_size


    def inference(self, inputs, scope='densenet121'):
        """
        Inception V3 net structure
        :param inputs:
        :param scope:
        :return:
        """
        self.prameter = []
        prop = self.densenet(inputs=inputs,
                             num_classes=self.num_classes,
                             keep_prob=self.keep_prob,
                             reduction=self.reduction,
                             grow_rate=self.grow_rate,
                             num_filters=self.num_filters,
                             num_dense_block=self.num_dense_block,
                             num_layers=self.num_layers,
                             reuse = self.reuse,
                             scope=scope,
                             is_training = self.is_training)
        return prop

    def densenet(self, inputs, scope='densenet121', num_classes=10, keep_prob=0.8, reduction=0.5, grow_rate=32,
                     num_filters=64, num_dense_block=4, num_layers=None, reuse=None, is_training=False):
        batch_norm_params = {
            'is_training': is_training,
            # Decay for the moving averages.
            'decay': self.batch_norm_decay,
            # epsilon to prevent 0s in variance.
            'epsilon': self.batch_norm_epsilon,
            # collection containing update_ops.
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            # use gamma for update
            'scale': self.batch_norm_scale,
            # use fused batch norm if possible.
            'fused': self.batch_norm_fused,
        }
        with tf.compat.v1.variable_scope(scope, default_name='densenet121', values=[inputs], reuse=reuse) as scope:
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                # Set weight_decay for weights in Conv and FC layers.
                with slim.arg_scope([slim.conv2d],
                                    weights_regularizer=slim.l2_regularizer(self.weight_decay),
                                    activation_fn=None,
                                    normalizer_fn=None,
                                    biases_initializer=None,
                                    ):
                    with slim.arg_scope([slim.batch_norm],
                                        decay = self.batch_norm_decay,
                                        epsilon = self.batch_norm_epsilon,
                                        updates_collections = tf.GraphKeys.UPDATE_OPS,
                                        # use gamma for update
                                        scale=self.batch_norm_scale,
                                        # use fused batch norm if possible.
                                        fused = self.batch_norm_fused
                                        ):
                        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d], stride=1, padding='VALID'):
                            # final block
                            net, num_filters = self.densenet_base(inputs=inputs, keep_prob=keep_prob,
                                                                      reduction=reduction,  growth_rate=grow_rate,
                                                                      num_filters=num_filters, num_layers=num_layers,
                                                                      num_dense_block=num_dense_block, scope=scope)
                            with tf.compat.v1.variable_scope('final_block', values=[net]):
                                net = slim.batch_norm(net)
                                net = tf.nn.relu(net, name='relu1')
                                # global_average pool
                                net = tf.reduce_mean(input_tensor=net, axis=[1, 2], keep_dims=True,
                                                     name='global_avg_pool1', )
                            # logits
                            logit = slim.conv2d(inputs=net, num_outputs=num_classes, kernel_size=(1, 1),
                                                stride=1, activation_fn=None, normalizer_fn=None,
                                                biases_initializer=tf.zeros_initializer(),scope='logits')
                            # squeeze
                            logit = tf.squeeze(input=logit, axis=[1, 2], name='spatial_squeeze')
                            prob = slim.softmax(logits=logit, scope='predict')
                            return prob


    def densenet_base(self, inputs, scope='densenet121', keep_prob=0.8, reduction=0.5,
                          growth_rate=32, num_filters=64, num_dense_block=4, num_layers=None):
        """
        inception V3 base
        :param inputs:
        :param scope:
        :return:
        """
        with tf.compat.v1.variable_scope(scope, default_name='densenet121', values=[inputs]):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='VALID'):
                # 224 x 224 x 3
                # zero padding 3
                inputs = tf.pad(inputs, paddings=[[0, 0], [3, 3], [3, 3], [0, 0]], name='padding_zeros1')
                # 230 x 230 x 3
                net = slim.conv2d(inputs=inputs, num_outputs=num_filters, kernel_size=(7, 7), stride=2, scope='conv1',
                                  padding='VALID')
                # 112 x 112 x 3
                net = slim.batch_norm(net)
                net = tf.nn.relu(features=net, name='relu1')
                # 112 x 112 x 3
                # zero padding
                net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], name='padding_zeros2')
                # 114 x 114 x 3
                net = slim.max_pool2d(inputs=net, kernel_size=(3, 3), stride=2, scope='max_pool1', padding='VALID')
                # 56 x 56 x 3
                for block_id in range(num_dense_block-1):
                    branch_id = block_id + 1
                    net, num_filters = self.dense_block(inputs=net, num_filters=num_filters,
                                                        num_layer=num_layers[block_id], growth_rate=growth_rate,
                                                        keep_prob=keep_prob, scope='dense_block'+str(branch_id))
                    net, num_filters = self.transition_block(inputs=net, num_filters=num_filters,
                                                             compression=(1-reduction), keep_prob=keep_prob,
                                                             scope='transition_block'+str(branch_id))

                net, num_filters = self.dense_block(inputs=net, num_filters=num_filters,
                                                    num_layer=num_layers[-1], growth_rate=growth_rate,
                                                    keep_prob=keep_prob, scope='dense_block'+str(num_dense_block))
                return net, num_filters

    def transition_block(self, inputs, num_filters, compression=1.0, keep_prob=1.0, scope=None):
        """
        transition block
        :param inputs:
        :param num_filters:
        :param compression:
        :param keep_prob:
        :return:
        """
        with tf.compat.v1.variable_scope(scope, default_name='transition_block', values=[inputs]):
            with tf.compat.v1.variable_scope('blk', default_name='blk'):
                compress_filters = int(num_filters * compression)
                net = slim.batch_norm(inputs)
                net = tf.nn.relu(features=net, name='relu1')
                compress_net = slim.conv2d(inputs=net, num_outputs=compress_filters, kernel_size=(1, 1), stride=1,
                                  padding='VALID')
                compress_net = slim.dropout(inputs=compress_net, keep_prob=keep_prob)
                # zero padding
                compress_net = slim.avg_pool2d(inputs=compress_net, kernel_size=(2, 2), stride=2, scope='avgpool1',
                                               padding='VALID')
                return compress_net, compress_filters

    def dense_block(self, inputs, num_filters, num_layer, growth_rate, keep_prob=1.0, growth_filters=True, scope=None):
        """
        dense_block
        :param inputs:
        :param num_filters:
        :param num_layer:
        :param growth_rate:
        :param grow_filters:
        :param scope:
        :return:
        """
        with tf.compat.v1.variable_scope(scope, default_name='dense_block', values=[inputs]):
            concat_net = inputs
            for n in range(num_layer):
                branch = n + 1
                net = self.bottleneck_block(inputs=concat_net, num_filters=growth_rate, keep_prob=keep_prob,
                                            scope='conv_block'+str(branch))
                concat_net = tf.concat(values=[concat_net, net], axis=3, name='dense_block'+str(branch)+'concat')
                if growth_filters:
                    num_filters += growth_rate
            return concat_net, num_filters

    def bottleneck_block(self, inputs, num_filters, keep_prob=1.0, scope=None):
        """
        bottleneck block
        :param inputs:
        :param stage:
        :param branch:
        :param num_filters:
        :param drop_rate:
        :param batch_norm_epsilon:
        :return:
        """
        with tf.compat.v1.variable_scope(scope, default_name='conv_block', values=[inputs]):
            with tf.compat.v1.variable_scope('x1', default_name='x1', values=[inputs]):
                # bottleneck layers
                inter_channel = num_filters * 4
                net = slim.batch_norm(inputs=inputs)
                net = tf.nn.relu(net, name='relu1')
                net = slim.conv2d(inputs=net, num_outputs=inter_channel, kernel_size=(1, 1), stride=1, padding='VALID')
                net = slim.dropout(inputs=net, keep_prob=keep_prob)
            with tf.compat.v1.variable_scope('x2', default_name='x2', values=[net]):
                # bottleneck layers
                net = slim.batch_norm(inputs=net)
                net = tf.nn.relu(net, name='relu2')
                # # zero padding
                # net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [1, 1]], name='padding_zeros2')
                net = slim.conv2d(inputs=net, num_outputs=num_filters, kernel_size=(3, 3), stride=1, padding='SAME')
                net = slim.dropout(inputs=net, keep_prob=keep_prob)
        return net

    def training(self, learnRate, globalStep, loss):
        """
        train operation
        :param learnRate:
        :param globalStep:
        :param args:
        :return:
        """

        # define trainable variable
        trainable_variable = None
        # trainable_scope = self.trainable_scope
        # trainable_scope = ['InceptionV3/Logits/Conv2d_1c_1x1']
        trainable_scope = []
        if self.is_pretrain and trainable_scope:
            trainable_variable = []
            for scope in trainable_scope:
                variables = tf.model_variables(scope=scope)
                [trainable_variable.append(var) for var in variables]

        learning_rate = tf.train.exponential_decay(learning_rate=learnRate, global_step=globalStep,
                                                   decay_steps=self.decay_steps, decay_rate=self.decay_rate,
                                                   staircase=False)
        # according to use request of slim.batch_norm
        # update moving_mean and moving_variance when training
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op =  tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=globalStep,
                                                                       var_list=trainable_variable)
        return train_op

    def losses(self, logits, labels, scope='Loss'):
        """
        loss function
        :param logits:
        :param labels:
        :return:
        """
        with tf.name_scope(scope) as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='Entropy')
            return tf.reduce_mean(input_tensor=cross_entropy, name='Entropy_Mean')

    def evaluate_batch(self, logits, labels, scope='Evaluate_Batch'):
        """
        evaluate one batch correct num
        :param logits:
        :param label:
        :return:
        """
        with tf.name_scope(scope):
            correct_predict = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            return tf.reduce_sum(tf.cast(correct_predict, dtype=tf.float32))

    def fill_feed_dict(self, image_feed, label_feed, is_training):
        feed_dict = {
            self.raw_input_data: image_feed,
            self.raw_input_label: label_feed,
            self.is_training: is_training
        }
        return feed_dict





