#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/2/2
"""Wide and Deep Model Prediction"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import sys
import time

import tensorflow as tf

from build_estimator import build_estimator
from dataset import Dataset
from lib.util import elapse_time
from read_conf import Config

CONFIG = Config().train
parser = argparse.ArgumentParser(description='Evaluate Wide and Deep Model.')

parser.add_argument(
    '--model_dir', type=str, default=CONFIG["model_dir"],
    help='Model checkpoint dir for evaluating.')

parser.add_argument(
    '--data_dir', type=str,
    help='Evaluating data dir.')

parser.add_argument(
    '--model_type', type=str, default=CONFIG["model_type"],
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")

parser.add_argument(
    '--batch_size', type=int, default=CONFIG["batch_size"],
    help='Number of examples per batch.')

parser.add_argument(
    '--checkpoint_path', type=str, default=CONFIG["checkpoint_path"],
    help="Path of a specific checkpoint to evaluate. If None, the latest checkpoint in model_dir is used.")


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

    checkpoint_path = FLAGS.checkpoint_path or model.latest_checkpoint()
    if checkpoint_path is None:
        raise ValueError('No model checkpoint found, please check the model dir.')
    tf.logging.info('Using model checkpoint: {}'.format(checkpoint_path))

    print('-' * 80)
    tf.logging.info('='*30+' START PREDICTION'+'='*30)
    t0 = time.time()
    predictions = model.predict(input_fn=lambda: Dataset().pred_input_fn(FLAGS.data_dir, FLAGS.batch_size),
                                predict_keys=None,
                                hooks=None,
                                checkpoint_path=checkpoint_path)  # defaults None to use latest_checkpoint
    tf.logging.info('='*30+'FINISH PREDICTION, TAKE {} mins'.format(elapse_time(t0))+'='*30)

    for pred_dict in predictions:  # dict{probabilities, classes, class_ids}
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        print('\nPrediction is "{}" ({:.1f}%), expected "{}"'.format(class_id, 100 * probability))

if __name__ == '__main__':
    # Set to INFO for tracking training, default is WARN. ERROR for least messages
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
