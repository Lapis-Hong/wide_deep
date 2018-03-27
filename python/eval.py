#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/15
"""Wide and Deep Model Evaluation"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import sys
import time

import tensorflow as tf

from lib.read_conf import Config
from lib.dataset import input_fn
from lib.build_estimator import build_estimator
from lib.utils.util import elapse_time

CONFIG = Config().train
parser = argparse.ArgumentParser(description='Evaluate Wide and Deep Model.')

parser.add_argument(
    '--model_dir', type=str, default=CONFIG["model_dir"],
    help='Model checkpoint dir for evaluating.')

parser.add_argument(
    '--model_type', type=str, default=CONFIG["model_type"],
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")

parser.add_argument(
    '--test_data', type=str, default=CONFIG["test_data"],
    help='Evaluating data dir.')

parser.add_argument(
    '--image_test_data', type=str, default=CONFIG["image_test_data"],
    help='Evaluating image data dir.')

parser.add_argument(
    '--batch_size', type=int, default=CONFIG["batch_size"],
    help='Number of examples per batch.')

parser.add_argument(
    '--checkpoint_path', type=str, default=CONFIG["checkpoint_path"],
    help="Path of a specific checkpoint to evaluate. If None, the latest checkpoint in model_dir is used.")

# TODO：support distributed evaluation or not ?
# parser.add_argument(
#     '--is_distribution', type=int, default=0,
#     help='Evaluating distributional or not')


def main(unused_argv):
    print("Using TensorFlow version %s" % tf.__version__)
    assert "1.4" <= tf.__version__, "TensorFlow r1.4 or later is needed"
    # if FLAGS.is_distribution:
    #     print("Using distribution tensoflow. Job_name:{} Task_index:{}"
    #           .format(CONFIG.distribution["job_name"], CONFIG.distribution["task_index"]))
    print('Model type: {}'.format(FLAGS.model_type))
    model_dir = os.path.join(FLAGS.model_dir, FLAGS.model_type)
    print('Model directory: {}'.format(model_dir))
    model = build_estimator(model_dir, FLAGS.model_type)
    tf.logging.info('Build estimator: {}'.format(model))
    # checkpoint_path = FLAGS.checkpoint_path or model.latest_checkpoint()
    # if checkpoint_path is None:
    #     raise ValueError('No model checkpoint found, please check the model dir.')
    # tf.logging.info('Using model checkpoint: {}'.format(checkpoint_path))
    # print('\n')
    tf.logging.info('='*30+' START TESTING'+'='*30)
    s_time = time.time()
    results = model.evaluate(input_fn=lambda: input_fn(FLAGS.test_data, FLAGS.image_test_data, 'eval', FLAGS.batch_size),
                             steps=None,  # Number of steps for which to evaluate model.
                             hooks=None,
                             checkpoint_path=FLAGS.checkpoint_path,  # If None, the latest checkpoint is used.
                             name=None)
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
