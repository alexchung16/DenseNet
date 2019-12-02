#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File ResNet50_train.py
# @ Description :
# @ Author alexchung
# @ Time 21/11/2019 PM 15:44

import os
import numpy as np
import tensorflow as tf
from DenseNet.DenseNet121_slim import DenseNet121
from TFRecordProcessing.parse_TFRecord import reader_tfrecord, get_num_samples
from tensorflow.python.framework import graph_util
#
original_dataset_dir = '/home/alex/Documents/datasets/flower_photos_separate'
tfrecord_dir = os.path.join(original_dataset_dir, 'tfrecord')

train_path = os.path.join(original_dataset_dir, 'train')
test_path = os.path.join(original_dataset_dir, 'test')
record_file = os.path.join(tfrecord_dir, 'image.tfrecords')
model_path = os.path.join(os.getcwd(), 'model')
model_name = os.path.join(model_path, 'densenet_121.pb')
pretrain_model_path = '/home/alex/Documents/pretraing_model/densenet/tf-densenet121'
pretrain_model = os.path.join(pretrain_model_path, 'tf-densenet121.ckpt')
logs_dir = os.path.join(os.getcwd(), 'logs')

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('height', 224, 'Number of height size.')
flags.DEFINE_integer('width', 224, 'Number of width size.')
flags.DEFINE_integer('depth', 3, 'Number of depth size.')
flags.DEFINE_integer('num_classes', 5, 'Number of image class.')
flags.DEFINE_integer('epoch', 30, 'Number of epoch size.')
flags.DEFINE_integer('step_per_epoch', 100, 'Number of step size of per epoch.')
flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')
flags.DEFINE_float('decay_rate', 0.99, 'Number of learning decay rate.')
# flags.DEFINE_bool('global_pool', False, 'if True, use global pool.')
# flags.DEFINE_bool('spacial_squeeze', True, 'if True, execute squeeze.')
flags.DEFINE_integer('num_epoch_per_decay', 2, 'Number epoch after each leaning rate decapy.')
flags.DEFINE_float('keep_prob', 0.5, 'Number of probability that each element is kept.')
flags.DEFINE_float('weight_decay', 0.0001, 'Number of weight decay with regular')

flags.DEFINE_float('reduction', 0.5, 'Number of reduction of transition block.')
flags.DEFINE_integer('grow_rate', 32, 'Number of filters to add per dense block.')
flags.DEFINE_integer('num_dense_block', 4, 'Number of block.')
flags.DEFINE_integer('num_filters', 64, 'Number of initial filters.')

flags.DEFINE_float('batch_norm_decay', 0.997, 'Number of batch norm decay.')
flags.DEFINE_float('batch_norm_epsilon', 1e-5, 'Number of batch norm epsilon.')
flags.DEFINE_bool('batch_norm_scale', False, 'if True, use gamma for update.')
flags.DEFINE_bool('batch_norm_fused', True, 'if True, use a faster, fused implementation if possible.')

flags.DEFINE_string('train_dir', record_file, 'Directory to put the training data.')
flags.DEFINE_bool('is_pretrain', True, 'if True, use pretrain model.')
flags.DEFINE_string('pretrain_model_dir', pretrain_model, 'pretrain model dir.')
flags.DEFINE_string('logs_dir', logs_dir, 'direct of summary logs.')
flags.DEFINE_string('model_name', model_name, 'model_name.')

def predict(model_name, image_data, input_op_name, predict_op_name):
    """
    model read and predict
    :param model_name:
    :param image_data:
    :param input_op_name:
    :param predict_op_name:
    :return:
    """
    with tf.Graph().as_default():
        with tf.gfile.FastGFile(name=model_name, mode='rb') as model_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model_file.read())
            _ = tf.import_graph_def(graph_def, name='')
        for index, layer in enumerate(graph_def.node):
            print(index, layer.name)

    with tf.Session() as sess:
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        sess.run(init_op)
        image = image_data.eval()
        input = sess.graph.get_tensor_by_name(name=input_op_name)
        output = sess.graph.get_tensor_by_name(name=predict_op_name)

        predict_softmax = sess.run(fetches=output, feed_dict={input: image})
        predict_label = np.argmax(predict_softmax, axis=1)
        return predict_label


# test build network
def networkStructureTest(batch_size):
    inputs = tf.random_uniform(shape=(batch_size, FLAGS.height, FLAGS.width, FLAGS.depth))
    labels = tf.random_uniform(shape=(batch_size, FLAGS.num_classes))

    with tf.Session() as sess:
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        sess.run(init_op)

        inputs, labels = sess.run([inputs, labels])
        feed_dict = densenet_121.fill_feed_dict(image_feed=inputs, label_feed=labels, is_training=True)

        logits = sess.run(fetches=[densenet_121.logits], feed_dict=feed_dict)
        assert list(logits[0].shape) == [batch_size, FLAGS.num_classes]

#
#
if __name__ == "__main__":

    num_layers = [6, 12, 24, 16]

    num_samples = get_num_samples(record_file=FLAGS.train_dir)
    batch_size = num_samples // FLAGS.step_per_epoch

    densenet_121 = DenseNet121(input_shape=[FLAGS.height, FLAGS.width, FLAGS.depth],
                            num_classes=FLAGS.num_classes,
                            batch_size=batch_size,
                            decay_rate=FLAGS.decay_rate,
                            learning_rate=FLAGS.learning_rate,
                            num_samples_per_epoch=num_samples,
                            num_epoch_per_decay=FLAGS.num_epoch_per_decay,
                            keep_prob=FLAGS.keep_prob,
                            reduction=FLAGS.reduction,
                            grow_rate=FLAGS.grow_rate,
                            num_filters=FLAGS.num_filters,
                            num_dense_block=FLAGS.num_dense_block,
                            num_layers=num_layers,
                            weight_decay=FLAGS.weight_decay,
                            batch_norm_decay=FLAGS.batch_norm_decay,
                            batch_norm_epsilon=FLAGS.batch_norm_epsilon,
                            batch_norm_fused=FLAGS.batch_norm_fused,
                            is_pretrain=FLAGS.is_pretrain)

    # add scalar value to summary protocol buffer
    tf.summary.scalar('loss', densenet_121.loss)
    tf.summary.scalar('accuracy', densenet_121.accuracy)

    # networkStructureTest(batch_size=batch_size)
    images, labels, filenames = reader_tfrecord(record_file=FLAGS.train_dir,
                                                batch_size=batch_size,
                                                input_shape=[FLAGS.height, FLAGS.width, FLAGS.depth],
                                                class_depth=FLAGS.num_classes,
                                                epoch=FLAGS.epoch,
                                                shuffle=False)
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    # train and save model

    with tf.Session() as sess:
        sess.run(init_op)

        # get computer graph
        graph = tf.get_default_graph()
        write = tf.summary.FileWriter(logdir=FLAGS.logs_dir, graph=graph)
        # get model variable of network
        model_variable = tf.model_variables()
        for var in model_variable:
            print(var.name, var.shape)

        # # get and add histogram to summary protocol buffer
        logit_weight = graph.get_tensor_by_name(name='densenet121/logits/weights:0')
        tf.summary.histogram(name='logits/weight', values=logit_weight)
        logit_biase = graph.get_tensor_by_name(name='densenet121/logits/biases:0')
        tf.summary.histogram(name='logits/biases', values=logit_biase)
        # merges all summaries collected in the default graph
        summary_op = tf.summary.merge_all()
        # load pretrain model
        if FLAGS.is_pretrain:
            # remove variable of fc8 layer from pretrain model
            custom_scope = ['densenet121/logits']
            for scope in custom_scope:
                variables = tf.model_variables(scope=scope)
                [model_variable.remove(var) for var in variables]
            saver = tf.train.Saver(var_list=model_variable)
            saver.restore(sess, save_path=FLAGS.pretrain_model_dir)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            if not coord.should_stop():
                for epoch in range(FLAGS.epoch):
                    print('Epoch: {0}/{1}'.format(epoch, FLAGS.epoch))
                    for step in range(FLAGS.step_per_epoch):

                        image, label, filename = sess.run([images, labels, filenames])

                        feed_dict = densenet_121.fill_feed_dict(image_feed=image,
                                                                label_feed=label,
                                                                is_training=True)

                        _, loss_value, train_accuracy, summary = sess.run(
                            fetches=[densenet_121.train, densenet_121.loss, densenet_121.accuracy, summary_op],
                            feed_dict=feed_dict)

                        print('  Step {0}/{1}: loss value {2}  train accuracy {3}'
                              .format(step, FLAGS.step_per_epoch, loss_value, train_accuracy))

                        write.add_summary(summary=summary, global_step=epoch*FLAGS.step_per_epoch+step)
                write.close()

                # save model
                # get op name for save model
                input_op = densenet_121.raw_input_data.name
                logit_op = densenet_121.logits.name
                # convert variable to constant
                def_graph = tf.get_default_graph().as_graph_def()
                constant_graph = tf.graph_util.convert_variables_to_constants(sess, def_graph,
                                                                              [input_op.split(':')[0],
                                                                               logit_op.split(':')[0]])
                # save to serialize file
                with tf.gfile.FastGFile(name=FLAGS.model_name, mode='wb') as f:
                    f.write(constant_graph.SerializeToString())

        except Exception as e:
            print(e)
        coord.request_stop()
    coord.join(threads)
    sess.close()
    print('model training has complete')

