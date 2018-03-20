#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/2/9
"""This module is based on tf.estimator.LinearClassifier.
linear logits builder for wide part"""
# TODO: add FM as linear part
import math

import tensorflow as tf

_LINEAR_LEARNING_RATE = 0.005


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
    return min(_LINEAR_LEARNING_RATE, default_learning_rate)


def linear_logit_fn_builder(units, feature_columns):
    """Function builder for a linear logit_fn.
    Args:
      units: An int indicating the dimension of the logit layer.
      feature_columns: An iterable containing all the feature columns used by the model.
    Returns:
      A logit_fn (see below).
    """

    def linear_logit_fn(features):
        """Linear model logit_fn.
        Args:
          features: This is the first item returned from the `input_fn`
                passed to `train`, `evaluate`, and `predict`. This should be a
                single `Tensor` or `dict` of same.
        Returns:
          A `Tensor` representing the logits.
        """
        return tf.feature_column.linear_model(
            units=units,
            features=features,
            feature_columns=feature_columns,
            sparse_combiner='sum',
            weight_collections=None,
            trainable=True,
        )

    return linear_logit_fn
