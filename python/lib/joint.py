#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/2/7
"""
TensorFlow Custom Estimators for Wide and Deep Joint Training Models.

There are two ways to build custom estimator.
    1. Write model_fn function to pass to `tf.estimator.Estimator` to generate an instance.
       easier to build but with less flexibility. 
    2. Write subclass of `tf.estimator.Estimator` like premade(canned) estimators.
       much suitable for official project. 
    
This module is based on tf.estimator.DNNLinearCombinedClassifier.
It merges `wide`, `deep`, `wide_deep` three types model into one class 
`WideAndDeepClassifier` by argument model_type 
It is a flexible and general joint learning framework.

Currently extensions:
    1. add BN layer options 
    2. arbitrary connections between layers (refer to ResNet and DenseNet)
    3. add Cnn as deep part 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf
from tensorflow.python.estimator.canned import head as head_lib

import os
import sys
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)

from lib.linear import linear_learning_rate, linear_logit_fn_builder
from lib.dnn import multidnn_logit_fn_builder
from lib.utils import model_util as util
from lib.cnn.vgg import Vgg16

# original import source
# from tensorflow.python.estimator import estimator
# from tensorflow.python.estimator.canned import dnn
# from tensorflow.python.estimator.canned import head as head_lib
# from tensorflow.python.estimator.canned import linear
# from tensorflow.python.estimator.canned import optimizers
# from tensorflow.python.framework import ops
# from tensorflow.python.ops import control_flow_ops
# from tensorflow.python.ops import nn
# from tensorflow.python.ops import partitioned_variables
# from tensorflow.python.ops import state_ops
# from tensorflow.python.ops import variable_scope
# from tensorflow.python.summary import summary
# from tensorflow.python.training import sync_replicas_optimizer
# from tensorflow.python.training import training_util

# The default learning rates are a historical artifact of the initial implementation.
_DNN_LEARNING_RATE = 0.001  # 0.05
_LINEAR_LEARNING_RATE = 0.005
_CNN_LEARNING_RATE = 0.01


def _wide_deep_combined_model_fn(
        features, labels, mode, head,
        model_type,
        with_cnn=False,
        cnn_optimizer='Adagrad',
        linear_feature_columns=None,
        linear_optimizer='Ftrl',
        dnn_feature_columns=None,
        dnn_optimizer='Adagrad',
        dnn_hidden_units=None,
        dnn_connected_mode=None,
        dnn_activation_fn=tf.nn.relu,
        dnn_dropout=None,
        dnn_batch_norm=None,
        input_layer_partitioner=None,
        config=None):
    """Wide and Deep combined model_fn. (Dnn, Cnn, Linear)
    Args:
        features: dict of `Tensor`.
        labels: `Tensor` of shape [batch_size, 1] or [batch_size] labels of dtype
            `int32` or `int64` in the range `[0, n_classes)`.
      mode: Defines whether this is training, evaluation or prediction. See `ModeKeys`.
      head: A `Head` instance.
      model_type: one of `wide`, `deep`, `wide_deep`.
      with_cnn: Bool, set True to combine image input featrues using cnn.
      cnn_optimizer: String, `Optimizer` object, or callable that defines the
        optimizer to use for training the CNN model. Defaults to the Adagrad
        optimizer.
      linear_feature_columns: An iterable containing all the feature columns used
          by the Linear model.
      linear_optimizer: String, `Optimizer` object, or callable that defines the
          optimizer to use for training the Linear model. Defaults to the Ftrl
          optimizer.
      dnn_feature_columns: An iterable containing all the feature columns used by
        the DNN model.
      dnn_optimizer: String, `Optimizer` object, or callable that defines the
        optimizer to use for training the DNN model. Defaults to the Adagrad
        optimizer.
      dnn_hidden_units: List of hidden units per DNN layer.
      dnn_connected_mode: List of connected mode.
      dnn_activation_fn: Activation function applied to each DNN layer. If `None`,
          will use `tf.nn.relu`.
      dnn_dropout: When not `None`, the probability we will drop out a given DNN
          coordinate.
      dnn_batch_norm: Bool, add BN layer after each DNN layer
      input_layer_partitioner: Partitioner for input layer.
          config: `RunConfig` object to configure the runtime settings.
    Returns:
        `ModelFnOps`
    Raises:
        ValueError: If both `linear_feature_columns` and `dnn_features_columns`
            are empty at the same time, or `input_layer_partitioner` is missing,
            or features has the wrong type.
    """
    if not isinstance(features, dict):
        raise ValueError('features should be a dictionary of `Tensor`s. '
                         'Given type: {}'.format(type(features)))
    if with_cnn:
        try:
            cnn_features = features.pop('image')  # separate image feature from input_fn
        except KeyError:
            raise ValueError('No input image features, must provide image features if use cnn.')
    num_ps_replicas = config.num_ps_replicas if config else 0
    input_layer_partitioner = input_layer_partitioner or (
        tf.min_max_variable_partitioner(max_partitions=num_ps_replicas,
                                        min_slice_size=64 << 20))
    # Build DNN Logits.
    dnn_parent_scope = 'dnn'
    if model_type == 'wide' or not dnn_feature_columns:
        dnn_logits = None
    else:
        dnn_optimizer = util.get_optimizer_instance(
            dnn_optimizer, learning_rate=_DNN_LEARNING_RATE)
        if model_type == 'wide_deep':
            util._check_no_sync_replicas_optimizer(dnn_optimizer)
        dnn_partitioner = tf.min_max_variable_partitioner(max_partitions=num_ps_replicas)
        with tf.variable_scope(
                dnn_parent_scope,
                values=tuple(six.itervalues(features)),
                partitioner=dnn_partitioner):
            dnn_logit_fn = multidnn_logit_fn_builder(
                units=head.logits_dimension,
                hidden_units_list=dnn_hidden_units,
                connected_mode_list=dnn_connected_mode,
                feature_columns=dnn_feature_columns,
                activation_fn=dnn_activation_fn,
                dropout=dnn_dropout,
                batch_norm=dnn_batch_norm,
                input_layer_partitioner=input_layer_partitioner
            )
            dnn_logits = dnn_logit_fn(features=features, mode=mode)

    # Build Linear Logits.
    linear_parent_scope = 'linear'
    if model_type == 'deep' or not linear_feature_columns:
        linear_logits = None
    else:
        linear_optimizer = util.get_optimizer_instance(linear_optimizer,
            learning_rate=linear_learning_rate(len(linear_feature_columns)))
        util._check_no_sync_replicas_optimizer(linear_optimizer)
        with tf.variable_scope(
                linear_parent_scope,
                values=tuple(six.itervalues(features)),
                partitioner=input_layer_partitioner) as scope:
            logit_fn = linear_logit_fn_builder(units=head.logits_dimension,
                feature_columns=linear_feature_columns)
            linear_logits = logit_fn(features=features)
            util.add_layer_summary(linear_logits, scope.name)

    # Build CNN Logits.
    cnn_parent_scope = 'cnn'
    if not with_cnn:
        cnn_logits = None
    else:
        cnn_optimizer = util.get_optimizer_instance(
            cnn_optimizer, learning_rate=_CNN_LEARNING_RATE)
        with tf.variable_scope(
                cnn_parent_scope,
                values=tuple([cnn_features]),
                partitioner=input_layer_partitioner) as scope:
            img_vec = Vgg16().build(cnn_features)
            cnn_logits = tf.layers.dense(
                img_vec,
                units=head.logits_dimension,
                kernel_initializer=tf.glorot_uniform_initializer(),
                name=scope)
            util.add_layer_summary(cnn_logits, scope.name)

    # Combine logits and build full model.
    logits_combine = []
    for logits in [dnn_logits, linear_logits, cnn_logits]:
        if logits is not None:
            logits_combine.append(logits)
    logits = tf.add_n(logits_combine)

    def _train_op_fn(loss):
        """Returns the op to optimize the loss."""
        train_ops = []
        global_step = tf.train.get_global_step()
        # BN, when training, the moving_mean and moving_variance need to be updated. By default the
        # update ops are placed in tf.GraphKeys.UPDATE_OPS, so they need to be added as a dependency to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if dnn_logits is not None:
                train_ops.append(
                    dnn_optimizer.minimize(
                        loss,
                        var_list=tf.get_collection(
                            tf.GraphKeys.TRAINABLE_VARIABLES,
                            scope=dnn_parent_scope)))
            if linear_logits is not None:
                train_ops.append(
                    linear_optimizer.minimize(
                        loss,
                        var_list=tf.get_collection(
                            tf.GraphKeys.TRAINABLE_VARIABLES,
                            scope=linear_parent_scope)))
            if cnn_logits is not None:
                train_ops.append(
                    cnn_optimizer.minimize(
                        loss,
                        var_list=tf.get_collection(
                            tf.GraphKeys.TRAINABLE_VARIABLES,
                            scope=cnn_parent_scope)))
            # Create an op that groups multiple ops. When this op finishes,
            # all ops in inputs have finished. This op has no output.
            train_op = tf.group(*train_ops)
        with tf.control_dependencies([train_op]):
            # Returns a context manager that specifies an op to colocate with.
            with tf.colocate_with(global_step):
                return tf.assign_add(global_step, 1)

    return head.create_estimator_spec(
                          features=features,
                          mode=mode,
                          labels=labels,
                          train_op_fn=_train_op_fn,
                          logits=logits)


class WideAndDeepClassifier(tf.estimator.Estimator):
    """An estimator for TensorFlow Wide and Deep joined classification models.
    Example:
    ```python
    numeric_feature = numeric_column(...)
    categorical_column_a = categorical_column_with_hash_bucket(...)
    categorical_column_b = categorical_column_with_hash_bucket(...)
    categorical_feature_a_x_categorical_feature_b = crossed_column(...)
    categorical_feature_a_emb = embedding_column(
        categorical_column=categorical_feature_a, ...)
    categorical_feature_b_emb = embedding_column(
        categorical_id_column=categorical_feature_b, ...)
    estimator = DNNLinearCombinedClassifier(
        # wide settings
        linear_feature_columns=[categorical_feature_a_x_categorical_feature_b],
        linear_optimizer=tf.train.FtrlOptimizer(...),
        # deep settings
        dnn_feature_columns=[
            categorical_feature_a_emb, categorical_feature_b_emb,
            numeric_feature],
        dnn_hidden_units=[1000, 500, 100],
        dnn_optimizer=tf.train.ProximalAdagradOptimizer(...))
    # To apply L1 and L2 regularization, you can set optimizers as follows:
    tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001,
        l2_regularization_strength=0.001)
    # It is same for FtrlOptimizer.
    # Input builders
    def input_fn_train: # returns x, y
        pass
    estimator.train(input_fn=input_fn_train, steps=100)
    def input_fn_eval: # returns x, y
        pass
    metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)
    def input_fn_predict: # returns x, None
        pass
    predictions = estimator.predict(input_fn=input_fn_predict)
    ```
    Input of `train` and `evaluate` should have following features,
    otherwise there will be a `KeyError`:
    * for each `column` in `dnn_feature_columns` + `linear_feature_columns`:
      - if `column` is a `_CategoricalColumn`, a feature with `key=column.name`
        whose `value` is a `SparseTensor`.
      - if `column` is a `_WeightedCategoricalColumn`, two features: the first
        with `key` the id column name, the second with `key` the weight column
        name. Both features' `value` must be a `SparseTensor`.
      - if `column` is a `_DenseColumn`, a feature with `key=column.name`
        whose `value` is a `Tensor`.
    Loss is calculated by using softmax cross entropy.
    @compatibility(eager)
    Estimators are not compatible with eager execution.
    @end_compatibility
    """
    def __init__(self,
                 model_type=None,
                 model_dir=None,
                 with_cnn=False,
                 cnn_optimizer='Adagrad',
                 linear_feature_columns=None,
                 linear_optimizer='Ftrl',
                 dnn_feature_columns=None,
                 dnn_optimizer='Adagrad',
                 dnn_hidden_units=None,
                 dnn_connected_mode=None,
                 dnn_activation_fn=tf.nn.relu,
                 dnn_dropout=None,
                 dnn_batch_norm=None,
                 n_classes=2,
                 weight_column=None,
                 label_vocabulary=None,
                 input_layer_partitioner=None,
                 config=None):
        """Initializes a WideDeepCombinedClassifier instance.
        Args:
            model_dir: Directory to save model parameters, graph and etc. This can
                also be used to load checkpoints from the directory into a estimator
                to continue training a previously saved model.
            linear_feature_columns: An iterable containing all the feature columns
                used by linear part of the model. All items in the set must be
                instances of classes derived from `FeatureColumn`.
            linear_optimizer: An instance of `tf.Optimizer` used to apply gradients to
                the linear part of the model. Defaults to FTRL optimizer.
            dnn_feature_columns: An iterable containing all the feature columns used
                by deep part of the model. All items in the set must be instances of
                classes derived from `FeatureColumn`.
            dnn_optimizer: An instance of `tf.Optimizer` used to apply gradients to
                the deep part of the model. Defaults to Adagrad optimizer.
            dnn_hidden_units: List of hidden units per layer. All layers are fully
                connected.
            dnn_activation_fn: Activation function applied to each layer. If None,
                will use `tf.nn.relu`.
            dnn_dropout: When not None, the probability we will drop out
                a given coordinate.
            n_classes: Number of label classes. Defaults to 2, namely binary
                classification. Must be > 1.
            weight_column: A string or a `_NumericColumn` created by
                `tf.feature_column.numeric_column` defining feature column representing
                weights. It is used to down weight or boost examples during training. It
                will be multiplied by the loss of the example. If it is a string, it is
                used as a key to fetch weight tensor from the `features`. If it is a
                `_NumericColumn`, raw tensor is fetched by key `weight_column.key`,
                then weight_column.normalizer_fn is applied on it to get weight tensor.
            label_vocabulary: A list of strings represents possible label values. If
                given, labels must be string type and have any value in
                `label_vocabulary`. If it is not given, that means labels are
                already encoded as integer or float within [0, 1] for `n_classes=2` and
                encoded as integer values in {0, 1,..., n_classes-1} for `n_classes`>2 .
                Also there will be errors if vocabulary is not provided and labels are
                string.
            input_layer_partitioner: Partitioner for input layer. Defaults to
                `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
            config: RunConfig object to configure the runtime settings.
        Raises:
            ValueError: If both linear_feature_columns and dnn_features_columns are
                empty at the same time.
        """
        if not linear_feature_columns and not dnn_feature_columns:
            raise ValueError('Either linear_feature_columns or dnn_feature_columns must be defined.')
        if model_type is None:
            raise ValueError("Model type must be defined. one of `wide`, `deep`, `wide_deep`.")
        else:
            assert model_type in {'wide', 'deep', 'wide_deep'}, (
                "Invalid model type, must be one of `wide`, `deep`, `wide_deep`.")
            if model_type == 'wide':
                if not linear_feature_columns:
                    raise ValueError('Linear_feature_columns must be defined for wide model.')
            elif model_type == 'deep':
                if not dnn_feature_columns:
                    raise ValueError('Dnn_feature_columns must be defined for deep model.')
        if dnn_feature_columns and not dnn_hidden_units:
            raise ValueError('dnn_hidden_units must be defined when dnn_feature_columns is specified.')

        if n_classes == 2:
            head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
                weight_column=weight_column,
                label_vocabulary=label_vocabulary)
        else:
            head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
                n_classes,
                weight_column=weight_column,
                label_vocabulary=label_vocabulary)

        def _model_fn(features, labels, mode, config):
            return _wide_deep_combined_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                head=head,
                model_type=model_type,
                with_cnn=with_cnn,
                cnn_optimizer=cnn_optimizer,
                linear_feature_columns=linear_feature_columns,
                linear_optimizer=linear_optimizer,
                dnn_feature_columns=dnn_feature_columns,
                dnn_connected_mode=dnn_connected_mode,
                dnn_optimizer=dnn_optimizer,
                dnn_hidden_units=dnn_hidden_units,
                dnn_activation_fn=dnn_activation_fn,
                dnn_dropout=dnn_dropout,
                dnn_batch_norm=dnn_batch_norm,
                input_layer_partitioner=input_layer_partitioner,
                config=config)
        super(WideAndDeepClassifier, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config)

