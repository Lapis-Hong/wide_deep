#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/2/2
"""Provide some utility function to the model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import time
from functools import wraps

import tensorflow as tf


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


"""Methods related to optimizers used in canned_estimators."""
_OPTIMIZER_CLS_NAMES = {
    'Adagrad': tf.train.AdagradOptimizer,
    'Adam': tf.train.AdamOptimizer,
    'Ftrl': tf.train.FtrlOptimizer,
    'RMSProp': tf.train.RMSPropOptimizer,
    'SGD': tf.train.GradientDescentOptimizer,
}


def get_optimizer_instance(opt, learning_rate=None):
    """Returns an optimizer instance.
    Supports the following types for the given `opt`:
        * An `Optimizer` instance: Returns the given `opt`.
        * A string: Creates an `Optimizer` subclass with the given `learning_rate`.
      Supported strings:
        * 'Adagrad': Returns an `AdagradOptimizer`.
        * 'Adam': Returns an `AdamOptimizer`.
        * 'Ftrl': Returns an `FtrlOptimizer`.
        * 'RMSProp': Returns an `RMSPropOptimizer`.
        * 'SGD': Returns a `GradientDescentOptimizer`.
    Args:
      opt: An `Optimizer` instance, or string, as discussed above.
      learning_rate: A float. Only used if `opt` is a string.
    Returns:
      An `Optimizer` instance.
    Raises:
      ValueError: If `opt` is an unsupported string.
      ValueError: If `opt` is a supported string but `learning_rate` was not specified.
      ValueError: If `opt` is none of the above types.
    """
    if isinstance(opt, six.string_types):
        if opt in six.iterkeys(_OPTIMIZER_CLS_NAMES):
            if not learning_rate:
                raise ValueError('learning_rate must be specified when opt is string.')
            return _OPTIMIZER_CLS_NAMES[opt](learning_rate=learning_rate)
        raise ValueError('Unsupported optimizer name: {}. Supported names are: {}'.format(
            opt, tuple(sorted(six.iterkeys(_OPTIMIZER_CLS_NAMES)))))
    if not isinstance(opt, tf.train.Optimizer):
        raise ValueError('The given object is not an Optimizer instance. Given: {}'.format(opt))
    return opt


"""An example of model_fn to build an custom estimator"""


def my_model(features, labels, mode, params):
    # TODO: add BN, add ResNet, add CNN
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def build_estimators(feature_columns):
    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10],
            # The model must choose between 3 classes.
            'n_classes': 3,
        })