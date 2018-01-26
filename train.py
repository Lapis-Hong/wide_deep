#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/15
"""Training Wide and Deep Model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import shutil
import sys
import time
import os
from functools import wraps

import tensorflow as tf

from read_conf import Config
from model import build_estimator
from input_fn import input_fn

CONFIG = Config().train
parser = argparse.ArgumentParser(description='Train Wide and Deep Model.')

parser.add_argument(
    '--model_dir', type=str, default=CONFIG["model_dir"],
    help='Base directory for the model.')
parser.add_argument(
    '--model_type', type=str, default=CONFIG["model_type"],
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")
parser.add_argument(
    '--train_epochs', type=int, default=CONFIG["train_epochs"],
    help='Number of training epochs.')
parser.add_argument(
    '--epochs_per_eval', type=int, default=1,
    help='The number of training epochs to run between evaluations.')
parser.add_argument(
    '--batch_size', type=int, default=CONFIG["batch_size"],
    help='Number of examples per batch.')
parser.add_argument(
    '--train_data', type=str, default=CONFIG["train_data"],
    help='Path to the training data.')
parser.add_argument(
    '--test_data', type=str, default=CONFIG["test_data"],
    help='Path to the test data.')
parser.add_argument(
    '--keep_train', type=int, default=CONFIG["keep_train"],
    help='Whether to keep training on previous trained model.')


def timer(info=''):
    """parameter decarotor"""
    def _timer(func):
        @wraps(func)
        def wrapper(*args, **kwargs):  # passing params to func
            s_time = time.time()
            func(*args, **kwargs)
            e_time = time.time()
            period = (e_time-s_time) / 60.0
            print(info + '-> elapsed time: %.3f minutes' % period)
        return wrapper
    return _timer


def elapse_time(start_time):
    return round((time.time()-start_time) / 60)


def main(unused_argv):
    CONFIG = Config()
    print("Using TensorFlow version %s" % tf.__version__)
    assert "1.4" <= tf.__version__, "TensorFlow r1.4 or later is needed"
    print("Using Train config: {}".format(CONFIG.train))
    if CONFIG.distribution["is_distribution"]:
        print("Using distribution tensoflow. Job_name:{} Task_index:{}"
              .format(CONFIG.distribution["job_name"], CONFIG.distribution["task_index"]))
    # model info
    tf.logging.info('Model type: {}'.format(FLAGS.model_type))
    model_dir = os.path.join(FLAGS.model_dir, FLAGS.model_type)
    tf.logging.info('Model directory: {}'.format(model_dir))
    # tf.gfile.MakeDirs(model_dir)  # tf.estimator auto create dir
    if not FLAGS.keep_train:
        # Clean up the model directory if present
        shutil.rmtree(model_dir, ignore_errors=True)
        tf.logging.info('Remove model directory: {}'.format(model_dir))
    model = build_estimator(model_dir, FLAGS.model_type)
    tf.logging.info('Build estimator: {}'.format(model))

    # check file exsits
    assert tf.gfile.Exists(FLAGS.train_data) and tf.gfile.Exists(FLAGS.test_data), (
       'train or test data file not found. Please make sure you have either default data_file or '
       'set both arguments --train_data and --test_data.')

    # if input file is a dir, convert to a file path list
    def list_files(input_data):
        if tf.gfile.IsDirectory(input_data):
            file_name = [f for f in tf.gfile.ListDirectory(input_data) if not f.startswith('.')]
            return [input_data + '/' + f for f in file_name]
        else:
            return [input_data]

    train_data_list = list_files(FLAGS.train_data)
    test_data_list = list_files(FLAGS.test_data)

    for n in range(FLAGS.train_epochs):
        print('-' * 80)
        tf.logging.info('='*30+' START EPOCH {} '.format(n+1)+'='*30)
        for f in train_data_list:
            s_time = time.time()
            tf.logging.info('<EPOCH {}>: Start training {}'.format(n+1, f))
            model.train(input_fn=lambda: input_fn(f, 1, FLAGS.batch_size, multivalue=False),
                        hooks=None,
                        steps=None,
                        max_steps=None,
                        saving_listeners=None
            )
            tf.logging.info('<EPOCH {}>: Finish training {}, take {} mins'.format(n+1, f, elapse_time(s_time)))
            print('-' * 80)
            tf.logging.info('<EPOCH {}>: Start evaluating {}'.format(n + 1, FLAGS.test_data))
            s_time = time.time()
            results = model.evaluate(input_fn=lambda: input_fn(test_data_list, 1, FLAGS.batch_size, False, multivalue=False),
                                     steps=None,  # Number of steps for which to evaluate model.
                                     hooks=None,
                                     checkpoint_path=None,  # If None, the latest checkpoint in model_dir is used.
                                     name=None
            )
            tf.logging.info('<EPOCH {}>: Finish evaluation {}, take {} mins'.format(n+1, FLAGS.test_data, elapse_time(s_time)))
            # train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000)
            # eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
            # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
            # Display evaluation metrics
            print('-' * 80)
            for key in sorted(results):
                print('%s: %s' % (key, results[key]))

    # # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
    # for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    #     print('-' * 80)
    #     tf.logging.info('Start training at epoch {}...'.format(n*FLAGS.epochs_per_eval+1))
    #     s_time = time.time()
    #     model.train(input_fn=lambda: input_fn(
    #         FLAGS.train_data, FLAGS.epochs_per_eval, FLAGS.batch_size, multivalue=False),
    #             hooks=None,
    #             steps=None,
    #             max_steps=None,
    #             saving_listeners=None
    #     )
    #     tf.logging.info('Finish {} epochs training, take {} mins'.format(FLAGS.epochs_per_eval, elapse_time(s_time)))
    #     print('-'*80)
    #     tf.logging.info('Start evaluating at epoch {}...'.format(n*FLAGS.epochs_per_eval+1))
    #     s_time = time.time()
    #     results = model.evaluate(input_fn=lambda: input_fn(
    #         FLAGS.test_data, 1, FLAGS.batch_size, False, multivalue=False),
    #                              steps=None,  # Number of steps for which to evaluate model.
    #                              hooks=None,
    #                              checkpoint_path=None,  # If None, the latest checkpoint in model_dir is used.
    #                              name=None
    #     )
    #     tf.logging.info('Finish evaluation, take {} mins'.format(elapse_time(s_time)))
    #     print('-'*80)
    #     # train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000)
    #     # eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
    #     # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    #     # Display evaluation metrics
    #     print('Results at epoch {}'.format((n + 1) * FLAGS.epochs_per_eval))
    #     print('-' * 80)
    #     for key in sorted(results):
    #         print('%s: %s' % (key, results[key]))


if __name__ == '__main__':
    # import tensorflow.contrib.eager as tfe
    # tfe.enable_eager_execution()
    # Set to INFO for tracking training, default is WARN. ERROR for least messages
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
