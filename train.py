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


def list_files(input_data):
    """if input file is a dir, convert to a file path list
    Return:
         file path list
    """
    if tf.gfile.IsDirectory(input_data):
        file_name = [f for f in tf.gfile.ListDirectory(input_data) if not f.startswith('.')]
        return [input_data + '/' + f for f in file_name]
    else:
        return [input_data]


def train():
    print("Using Train config: {}".format(CONFIG.train))
    # model info
    print('Model type: {}'.format(FLAGS.model_type))
    model_dir = os.path.join(FLAGS.model_dir, FLAGS.model_type)
    print('Model directory: {}'.format(model_dir))
    # tf.gfile.MakeDirs(model_dir)  # tf.estimator auto create dir
    if not FLAGS.keep_train:
        # Clean up the model directory if present
        shutil.rmtree(model_dir, ignore_errors=True)
        print('Remove model directory: {}'.format(model_dir))
    model = build_estimator(model_dir, FLAGS.model_type)
    tf.logging.info('Build estimator: {}'.format(model))

    # check file exsits
    assert tf.gfile.Exists(FLAGS.train_data) and tf.gfile.Exists(FLAGS.test_data), (
        'train or test data file not found. Please make sure you have either default data_file or '
        'set both arguments --train_data and --test_data.')

    train_data_list = list_files(FLAGS.train_data)

    for n in range(FLAGS.train_epochs):
        print()
        tf.logging.info('=' * 30 + ' START EPOCH {} '.format(n + 1) + '=' * 30 + '\n')
        for f in train_data_list:
            s_time = time.time()
            tf.logging.info('<EPOCH {}>: Start training {}'.format(n + 1, f))
            model.train(input_fn=lambda: input_fn(f, 1, FLAGS.batch_size, multivalue=False),
                        hooks=None,
                        steps=None,
                        max_steps=None,
                        saving_listeners=None
                        )
            tf.logging.info('<EPOCH {}>: Finish training {}, take {} mins'.format(n + 1, f, elapse_time(s_time)))
            print('-' * 80)
            tf.logging.info('<EPOCH {}>: Start evaluating {}'.format(n + 1, FLAGS.test_data))
            s_time = time.time()
            results = model.evaluate(
                input_fn=lambda: input_fn(FLAGS.test_data, 1, FLAGS.batch_size, False, multivalue=False),
                steps=None,  # Number of steps for which to evaluate model.
                hooks=None,
                checkpoint_path=None,  # If None, the latest checkpoint in model_dir is used.
                name=None
                )
            tf.logging.info(
                '<EPOCH {}>: Finish evaluation {}, take {} mins'.format(n + 1, FLAGS.test_data, elapse_time(s_time)))
            # train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000)
            # eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
            # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
            # Display evaluation metrics
            print('-' * 80)
            for key in sorted(results):
                print('%s: %s' % (key, results[key]))


# def _create_experiment_fn():
#     """Experiment creation function."""
#     # Get configuration from environment variables, implementation of tf.estimator.RunConfig interface.
#     # run_config = tf.contrib.learn.RunConfig()
#     estimator = build_estimator(FLAGS.model_dir, FLAGS.model_type)
#     return tf.contrib.learn.Experiment(
#         estimator=estimator,
#         train_input_fn=lambda: input_fn(FLAGS.train_data, 1, FLAGS.batch_size, multivalue=False),
#         eval_input_fn=lambda: input_fn(FLAGS.test_data, 1, FLAGS.batch_size, False, multivalue=False),
#         train_steps=1000,  # Perform this many steps of training. None, the default, means train forever.
#         eval_steps=None  # evaluate runs until input is exhausted (or another exception is raised), or for eval_steps steps, if specified.
#     )


def main(unused_argv):
    print("Using TensorFlow version %s" % tf.__version__)
    assert "1.4" <= tf.__version__, "Need TensorFlow r1.4 or later."

    if CONFIG.distribution["is_distribution"]:
        print("Using PID: {}".format(os.getpid()))
        cluster = CONFIG.distribution["cluster"]
        job_name = CONFIG.distribution["job_name"]
        task_index = CONFIG.distribution["task_index"]
        print("Using Distributed TensorFlow. Local host: {} Job_name: {} Task_index: {}"
              .format(cluster[job_name][task_index], job_name, task_index))
        cluster = tf.train.ClusterSpec(CONFIG.distribution["cluster"])
        server = tf.train.Server(cluster,
                                 job_name=job_name,
                                 task_index=task_index)
        if job_name == 'ps':
            # wait for incoming connection forever
            server.join()
            # sess = tf.Session(server.target)
            # queue = create_done_queue(task_index, num_workers)
            # for i in range(num_workers):
            #     sess.run(queue.dequeue())
            #     print("ps {} received worker {} done".format(task_index, i)
            # print("ps {} quitting".format(task_index))
        else:
            # TODO：supervisor & MonotoredTrainingSession & experiment
            # Each worker only needs to contact the PS task(s) and the local worker task.
            # config = tf.ConfigProto(device_filters=[
            #     '/job:ps', '/job:worker/task:%d' % arguments.task_index])
            with tf.device(tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % task_index,
                    cluster=cluster)):
                train()
            # e = _create_experiment_fn()
            # e.train_and_evaluate()  # call estimator's train() and evaluate() method
            hooks = [tf.train.StopAtStepHook(last_step=10000)]

            # with tf.train.MonitoredTrainingSession(
            #         master=server.target,
            #         is_chief=(task_index == 0),
            #         checkpoint_dir=args.model_dir,
            #         hooks=hooks
            # ) as mon_sess:
            #     while not mon_sess.should_stop():
            #         # mon_sess.run()
            #         classifier.fit(input_fn=train_input_fn, steps=1)

    else:
        train()


if __name__ == '__main__':
    # import tensorflow.contrib.eager as tfe
    # tfe.enable_eager_execution()
    # Set to INFO for tracking training, default is WARN. ERROR for least messages
    CONFIG = Config()
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
