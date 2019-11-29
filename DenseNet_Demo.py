#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File DenseNet_Demo.py
# @ Description :
# @ Author alexchung
# @ Time 28/11/2019 PM 20:15


import os
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf

import numpy as np
from TFRecordProcessing.parse_TFRecord import reader_tfrecord, get_num_samples
from tensorflow.python.framework import graph_util


original_dataset_dir = '/home/alex/Documents/datasets/dogs_vs_cat_separate'
tfrecord_dir = os.path.join(original_dataset_dir, 'tfrecord')

train_path = os.path.join(original_dataset_dir, 'train')
test_path = os.path.join(original_dataset_dir, 'test')
record_file = os.path.join(tfrecord_dir, 'image.tfrecords')

model_path = '/home/alex/Documents/pretraing_model/densenet/tf-densenet121'
pretrain_model_dir = os.path.join(model_path, 'tf-densenet121.ckpt')


if __name__ == "__main__":
    # restore model parameter
    with tf.Session() as sess:
        # sess.run(init)
        # print(sess.run(w1))

        new_saver = tf.train.import_meta_graph(os.path.join(model_path, 'tf-densenet121.ckpt.meta'))
        new_saver.restore(sess, save_path=pretrain_model_dir)
        for var in tf.model_variables():
            print(var.name)
