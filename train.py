#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/15
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys
import time
import os
from functools import wraps

import tensorflow as tf
from model import build_estimator, input_fn, _parse_csv, _parse_csv2


class Config():
    """Parser for train.conf"""
    # parse config
    config = {}
    for line in open('./conf/train.conf'):
        line = line.strip().strip('\n')
        if line.startswith('#') or not line:
            continue
        key = line.split('=')[0].strip().lower()
        value = line.split('=')[1].strip().lower()
        config[key] = value
    # train parameter
    model_dir = config["model_dir"]
    model_type = config["model_type"]
    train_data = config["train_data"]
    test_data = config["test_data"]
    train_epochs = config["train_epochs"]
    batch_size = int(config["batch_size"])
    keep_train = bool(config["keep_train"])
    is_distribution = bool(config["is_distribution"])
    # model hyperparameter
    dnn_config = map(int, config["dnn_config"].split(','))
    wide_l1 = float(config["wide_l1"])
    wide_l2 = float(config["wide_l2"])
    dnn_l1 = float(config["dnn_l1"])
    dnn_l2 = float(config["dnn_l2"])
    dnn_dropout = float(config["dnn_dropout"])




parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default='./model',
    help='Base directory for the model.')
parser.add_argument(
    '--model_type', type=str, default='wide_deep',
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")
parser.add_argument(
    '--train_epochs', type=int, default=5, help='Number of training epochs.')
parser.add_argument(
    '--epochs_per_eval', type=int, default=1,
    help='The number of training epochs to run between evaluations.')
parser.add_argument(
    '--batch_size', type=int, default=256, help='Number of examples per batch.')
parser.add_argument(
    '--train_data', type=str, default='./data/train',
    help='Path to the training data.')
parser.add_argument(
    '--test_data', type=str, default='./data/test',
    help='Path to the test data.')
parser.add_argument(
    '--keep_train', type=int, default=0,
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
    print("Using TensorFlow version %s" % tf.__version__)
    # model info
    model_type = FLAGS.model_type
    tf.logging.info('Model type: {}'.format(model_type))
    base_dir = FLAGS.model_dir
    model_dir = os.path.join(base_dir, model_type)
    tf.logging.info('Model directory: {}'.format(model_dir))
    # tf.gfile.MakeDirs(model_dir)  # tf.estimator auto create dir
    if not FLAGS.keep_train:
        # Clean up the model directory if present
        shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
    model = build_estimator(model_dir, model_type)

    # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
    for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        print('-' * 80)
        tf.logging.info('Start training at epoch {}...'.format(n*FLAGS.epochs_per_eval+1))
        s_time = time.time()
        model.train(input_fn=lambda: input_fn(
            FLAGS.train_data, _parse_csv2, FLAGS.epochs_per_eval, FLAGS.batch_size),
                hooks=None,
                steps=None,
                max_steps=None,
                saving_listeners=None
        )
        tf.logging.info('Finish {} epochs training, take {} mins'.format(FLAGS.epochs_per_eval, elapse_time(s_time)))
        print('-'*80)
        tf.logging.info('Start evaluating at epoch {}...'.format(n*FLAGS.epochs_per_eval+1))
        s_time = time.time()
        results = model.evaluate(input_fn=lambda: input_fn(
            FLAGS.test_data, _parse_csv2, 1, FLAGS.batch_size, False),
                                 steps=None,  # Number of steps for which to evaluate model.
                                 hooks=None,
                                 checkpoint_path=None,  # If None, the latest checkpoint in model_dir is used.
                                 name=None
        )
        tf.logging.info('Finish evaluation, take {} mins'.format(elapse_time(s_time)))
        print('-'*80)
        # train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000)
        # eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
        # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        # Display evaluation metrics
        print('Results at epoch {}'.format((n + 1) * FLAGS.epochs_per_eval))
        print('-' * 80)
        for key in sorted(results):
            print('%s: %s' % (key, results[key]))


if __name__ == '__main__':
    # import tensorflow.contrib.eager as tfe
    # tfe.enable_eager_execution()
    # Set to INFO for tracking training, default is WARN. ERROR for least messages
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
