#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/15
"""Evaluate Wide and Deep Model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import sys
import os
import time

import tensorflow as tf

from read_conf import Config
from model import build_estimator
from input_fn import input_fn

CONFIG = Config().test
parser = argparse.ArgumentParser(description='Evaluate Wide and Deep Model.')

parser.add_argument(
    '--model_dir', type=str, default=CONFIG["model_dir"],
    help='Model checkpoint dir for evaluating.')

parser.add_argument(
    '--data_dir', type=str, default=CONFIG["data_dir"],
    help='Evaluating data dir.')

parser.add_argument(
    '--model_type', type=str, default=CONFIG["model_type"],
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")

parser.add_argument(
    '--batch_size', type=int, default=CONFIG["batch_size"],
    help='Number of examples per batch.')

parser.add_argument(
    '--steps', type=str, default=CONFIG["steps"] or None,
    help="Number of steps for which to evaluate model. "
         "If None, evaluates until input_fn raises an end-of-input exception.")

parser.add_argument(
    '--check_point_path', type=str, default=CONFIG["check_point_path"] or None,
    help="Path of a specific checkpoint to evaluate. If None, the latest checkpoint in model_dir is used.")

# # TODO：
# parser.add_argument(
#     '--is_distribution', type=int, default=0,
#     help='Evaluating distributional or not')


def elapse_time(start_time):
    return round((time.time()-start_time) / 60)


def main(unused_argv):
    print("Using TensorFlow version %s" % tf.__version__)
    assert "1.4" <= tf.__version__, "TensorFlow r1.4 or later is needed"
    # if FLAGS.is_distribution:
    #     print("Using distribution tensoflow. Job_name:{} Task_index:{}"
    #           .format(CONFIG.distribution["job_name"], CONFIG.distribution["task_index"]))
    # model info
    print('Model type: {}'.format(FLAGS.model_type))
    model_dir = os.path.join(FLAGS.model_dir, FLAGS.model_type)
    print('Model directory: {}'.format(model_dir))
    model = build_estimator(model_dir, FLAGS.model_type)
    tf.logging.info('Build estimator: {}'.format(model))
    checkpoint_path = FLAGS.check_point_path or model.latest_checkpoint()
    if checkpoint_path is None:
        raise ValueError('No model checkpoint found, please check the model dir.')
    tf.logging.info('Using model checkpoint: {}'.format(checkpoint_path))

    # check file exsits
    assert tf.gfile.Exists(FLAGS.data_dir), (
       'data file not found. Please make sure you have either default data_file or '
       'set arguments --data_dir.')

    # if input file is a dir, convert to a file path list
    def list_files(input_data):
        if tf.gfile.IsDirectory(input_data):
            file_name = [f for f in tf.gfile.ListDirectory(input_data) if not f.startswith('.')]
            return [input_data + '/' + f for f in file_name]
        else:
            return [input_data]
    data_list = list_files(FLAGS.data_dir)

    print('-' * 80)
    tf.logging.info('='*30+' START TESTING'+'='*30)
    s_time = time.time()
    results = model.evaluate(input_fn=lambda: input_fn(data_list, 1, FLAGS.batch_size, False, multivalue=False),
                             steps=None,  # Number of steps for which to evaluate model.
                             hooks=None,
                             checkpoint_path=FLAGS.check_point_path,  # If None, the latest checkpoint in model_dir is used.
                             name=None
    )
    tf.logging.info('='*30+'FINISH TESTING, TAKE {}'.format(elapse_time(s_time))+'='*30)
    # Display evaluation metrics
    print('-' * 80)
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))

if __name__ == '__main__':
    # Set to INFO for tracking training, default is WARN. ERROR for least messages
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
