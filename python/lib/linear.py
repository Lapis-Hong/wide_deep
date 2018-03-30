#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/2/9
"""This module is based on tf.estimator.LinearClassifier.
linear logits builder for wide part"""
# TODO: add FM as linear part
import tensorflow as tf


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
