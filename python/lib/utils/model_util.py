#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/3/5
"""This module contains some model build related utility functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import math
import tensorflow as tf


def add_layer_summary(value, tag):
    tf.summary.scalar('%s/fraction_of_zero_values' % tag, tf.nn.zero_fraction(value))
    tf.summary.histogram('%s/activation' % tag, value)


def check_no_sync_replicas_optimizer(optimizer):
    if isinstance(optimizer, tf.train.SyncReplicasOptimizer):
        raise ValueError(
            'SyncReplicasOptimizer does not support multi optimizers case. '
            'Therefore, it is not supported in DNNLinearCombined model. '
            'If you want to use this optimizer, please use either DNN or Linear model.')


def activation_fn(opt):
    """Returns an activation function.
    Args:
        opt: string
        Supported 10 strings:
        * 'sigmoid': Returns `tf.sigmoid`.
        * 'tanh': Returns `tf.tanh`.
        * 'relu': Returns `tf.nn.relu`.
        * 'relu6': Returns `tf.nn.relu6`.
        * 'leaky_relu': Returns `tf.nn.leaky_relu`.
        * 'crelu': Returns `tf.nn.crelu`.
        * 'elu': Returns `tf.nn.elu`.
        * 'selu': Returns `tf.nn.selu`.
        * 'softplus': Returns `tf.nn.softplus`.
        * 'softsign': Returns `tf.nn.softsign`.
    """
    _activation_fn_name = {
        'sigmoid': tf.sigmoid,
        'tanh': tf.tanh,
        'relu': tf.nn.relu,
        'relu6': tf.nn.relu6,
        'leaky_relu': tf.nn.leaky_relu,
        'crelu': tf.nn.crelu,
        'elu': tf.nn.elu,
        'selu': tf.nn.selu,
        'softplus': tf.nn.softplus,
        'softsign': tf.nn.softsign,
    }
    if opt in six.iterkeys(_activation_fn_name):
        return _activation_fn_name[opt]
    raise ValueError('Unsupported activation name: {}. Supported names are: {}'.format(
        opt, tuple(sorted(six.iterkeys(_activation_fn_name)))))


def get_optimizer_instance(opt, learning_rate=None):
    """Returns an optimizer instance.
    Supports the following types for the given `opt`:
        * An `Optimizer` instance string: Returns the given `opt`.
        * A supported string: Creates an `Optimizer` subclass with the given `learning_rate`.
      Supported strings:
        * 'Adagrad': Returns an `AdagradOptimizer`.
        * 'Adam': Returns an `AdamOptimizer`.
        * 'Ftrl': Returns an `FtrlOptimizer`.
        * 'RMSProp': Returns an `RMSPropOptimizer`.
        * 'SGD': Returns a `GradientDescentOptimizer`.
    Args:
      opt: An `Optimizer` instance, or supported string, as discussed above.
      learning_rate: A float. Only used if `opt` is a supported string.
    Returns:
      An `Optimizer` instance.
    Raises:
      ValueError: If `opt` is an unsupported string.
      ValueError: If `opt` is a supported string but `learning_rate` was not specified.
      ValueError: If `opt` is none of the above types.
    """
    # Methods related to optimizers used in canned_estimators."""
    _OPTIMIZER_CLS_NAMES = {
        'Adagrad': tf.train.AdagradOptimizer,
        'Adam': tf.train.AdamOptimizer,
        'Ftrl': tf.train.FtrlOptimizer,
        'RMSProp': tf.train.RMSPropOptimizer,
        'SGD': tf.train.GradientDescentOptimizer
    }
    if isinstance(opt, six.string_types):
        if opt in six.iterkeys(_OPTIMIZER_CLS_NAMES):
            if learning_rate is None:
                raise ValueError('learning_rate must be specified when opt is supported string.')
            return _OPTIMIZER_CLS_NAMES[opt](learning_rate=learning_rate)
        else:
            try:
                opt = eval(opt)  # eval('tf.nn.relu') tf.nn.relu
                if not isinstance(opt, tf.train.Optimizer):
                    raise ValueError('The given object is not an Optimizer instance. Given: {}'.format(opt))
                return opt
            except (AttributeError, NameError):
                raise ValueError('Unsupported optimizer option: `{}`. '
                    'Supported names are: {} or an `Optimizer` instance.'.format(
                    opt, tuple(sorted(six.iterkeys(_OPTIMIZER_CLS_NAMES)))))


def linear_learning_rate(num_linear_feature_columns):
    """Returns the default learning rate of the linear model.
    The calculation is a historical artifact of this initial implementation, but
    has proven a reasonable choice.
    Args:
      num_linear_feature_columns: The number of feature columns of the linear model.
    Returns:
      A float.
    """
    default_learning_rate = 1. / math.sqrt(num_linear_feature_columns)
    return min(0.005, default_learning_rate)