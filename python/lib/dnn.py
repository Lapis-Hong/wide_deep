#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/2/9
"""This module is based on tf.estimator.DNNClassifier.
Dnn logits builder. 
Extend dnn architecture, add BN layer, add Regularization, add activation function options, add arbitrary connections between layers.
Extend dnn to multi joint dnn.
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

from lib.read_conf import Config
from lib.utils.model_util import add_layer_summary, get_optimizer_instance, activation_fn

CONF = Config().model
ACTIVATION_FN = activation_fn(CONF['dnn_activation_function'])
DROPOUT = CONF['dnn_dropout']
BATCH_NORM = CONF['dnn_batch_normalization']
DNN_L1 = CONF['dnn_l1']
DNN_L2 = CONF['dnn_l2']
regularizer_list = []
if DNN_L1:
    regularizer_list.append(tf.contrib.layers.l1_regularizer(DNN_L1))
if DNN_L2:
    regularizer_list.append(tf.contrib.layers.l2_regularizer(DNN_L2))
if len(regularizer_list) == 0:
    REG = None
else:
    REG = tf.contrib.layers.sum_regularizer(regularizer_list)


def _dnn_logit_fn(features, mode, model_id, units,
                  hidden_units, connected_mode, feature_columns, input_layer_partitioner):
    """Deep Neural Network logit_fn.
    Args:
        features: This is the first item returned from the `input_fn`
            passed to `train`, `evaluate`, and `predict`. This should be a
            single `Tensor` or `dict` of same.
        mode: Optional. Specifies if this training, evaluation or prediction. See
            `ModeKeys`.
        model_id: An int indicating the model index of multi dnn.
        units: An int indicating the dimension of the logit layer.  In the
            MultiHead case, this should be the sum of all component Heads' logit
            dimensions.
        hidden_units: Iterable of integer number of hidden units per layer.
        connected_mode: one of {`simple`, `first_dense`, `last_dense`, `dense`, `resnet`}
            or arbitrary connections index tuples.
            1. `simple`: normal dnn architecture.
            2. `first_dense`: add addition connections from first input layer to all hidden layers.
            3. `last_dense`: add addition connections from all previous layers to last layer.
            4. `dense`: add addition connections between all layers, similar to DenseNet.
            5. `resnet`: add addition connections between adjacent layers, similar to ResNet.
            6. arbitrary connections list: add addition connections from layer_0 to layer_1 like 0-1.
                eg: [0-1,0-3,1-2]  index start from zero (input_layer), max index is len(hidden_units), smaller index first.
        feature_columns: Iterable of `feature_column._FeatureColumn` model inputs.
        activation_fn: Activation function applied to each layer.
        dropout: When not `None`, the probability we will drop out a given coordinate.
        batch_norm: Bool, Whether to use BN in dnn.
        input_layer_partitioner: Partitioner for input layer.
    Returns:
        A `Tensor` representing the logits, or a list of `Tensor`'s representing
      multiple logits in the MultiHead case.
    Raises:
        AssertError: If connected_mode is string, but not one of `simple`, `first_dense`, `last_dense`, 
            `dense` or `resnet`
    """

    if isinstance(connected_mode, str):
        assert connected_mode in {'simple', 'first_dense', 'lase_dense', 'dense', 'resnet'}, (
            'Invalid connected_mode: {}'.format(connected_mode)
        )
    with tf.variable_scope(
            'input_from_feature_columns',
            values=tuple(six.itervalues(features)),
            partitioner=input_layer_partitioner,
            reuse=tf.AUTO_REUSE):
        net = tf.feature_column.input_layer(
            features=features,
            feature_columns=feature_columns)
    input_layer = net
    if connected_mode == 'simple':
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with tf.variable_scope('dnn_{}/hiddenlayer_{}'.format(model_id, layer_id),
                    values=(net,)) as hidden_layer_scope:
                net = tf.layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=ACTIVATION_FN,
                    use_bias=True,
                    kernel_initializer=tf.glorot_uniform_initializer(),  # also called Xavier uniform initializer.
                    bias_initializer=tf.zeros_initializer(),
                    kernel_regularizer=REG,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    bias_constraint=None,
                    trainable=True,
                    reuse=None,
                    name=hidden_layer_scope)
                if DROPOUT is not None and mode == tf.estimator.ModeKeys.TRAIN:
                    net = tf.layers.dropout(net, rate=DROPOUT, training=True)  # rate=0.1 would drop out 10% of input units.
                if BATCH_NORM:
                    net = tf.layers.batch_normalization(net)
            add_layer_summary(net, hidden_layer_scope.name)

    elif connected_mode == 'first_dense':
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with tf.variable_scope('dnn_{}/hiddenlayer_{}'.format(model_id, layer_id),
                    values=(net,)) as hidden_layer_scope:
                net = tf.layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=ACTIVATION_FN,
                    kernel_initializer=tf.glorot_uniform_initializer(),  # also called Xavier uniform initializer.
                    kernel_regularizer=REG,
                    name=hidden_layer_scope)
                if DROPOUT is not None and mode == tf.estimator.ModeKeys.TRAIN:
                    net = tf.layers.dropout(net, rate=DROPOUT, training=True)
                if BATCH_NORM:
                    net = tf.layers.batch_normalization(net)
                net = tf.concat([net, input_layer], axis=1)
            add_layer_summary(net, hidden_layer_scope.name)

    elif connected_mode == 'last_dense':
        net_collections = [input_layer]
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with tf.variable_scope('dnn_{}/hiddenlayer_{}'.format(model_id, layer_id),
                    values=(net,)) as hidden_layer_scope:
                net = tf.layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=ACTIVATION_FN,
                    kernel_initializer=tf.glorot_uniform_initializer(),  # also called Xavier uniform initializer.
                    kernel_regularizer=REG,
                    name=hidden_layer_scope)
                if DROPOUT is not None and mode == tf.estimator.ModeKeys.TRAIN:
                    net = tf.layers.dropout(net, rate=DROPOUT, training=True)
                if BATCH_NORM:
                    net = tf.layers.batch_normalization(net)
                net_collections.append(net)
            add_layer_summary(net, hidden_layer_scope.name)
        net = tf.concat(net_collections, axis=1)  # Concatenates the list of tensors `values` along dimension `axis`

    elif connected_mode == 'dense':
        net_collections = [input_layer]
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with tf.variable_scope('dnn_{}/hiddenlayer_{}'.format(model_id, layer_id),
                    values=(net,)) as hidden_layer_scope:
                net = tf.layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=ACTIVATION_FN,
                    kernel_initializer=tf.glorot_uniform_initializer(),  # also called Xavier uniform initializer.
                    kernel_regularizer=REG,
                    name=hidden_layer_scope)
                if DROPOUT is not None and mode == tf.estimator.ModeKeys.TRAIN:
                    net = tf.layers.dropout(net, rate=DROPOUT, training=True)  # rate=0.1 would drop out 10% of input units.
                if BATCH_NORM:
                    net = tf.layers.batch_normalization(net)
                net_collections.append(net)
                net = tf.concat(net_collections, axis=1)
            add_layer_summary(net, hidden_layer_scope.name)

    elif connected_mode == 'resnet':  # connect layers in turn 0-1; 1-2; 2-3;
        net_collections = [input_layer]
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with tf.variable_scope('dnn_{}/hiddenlayer_{}'.format(model_id, layer_id),
                    values=(net,)) as hidden_layer_scope:
                net = tf.layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=ACTIVATION_FN,
                    kernel_initializer=tf.glorot_uniform_initializer(),  # also called Xavier uniform initializer.
                    kernel_regularizer=REG,
                    name=hidden_layer_scope)
                if DROPOUT is not None and mode == tf.estimator.ModeKeys.TRAIN:
                    net = tf.layers.dropout(net, rate=DROPOUT, training=True)
                if BATCH_NORM:
                    net = tf.layers.batch_normalization(net)
                net = tf.concat([net, net_collections[layer_id + 1 - 1]], axis=1)
                net_collections.append(net)
            add_layer_summary(net, hidden_layer_scope.name)

    else:  # arbitrary connections, ['0-1','0-3','1-3'], small index layer first
        connected_mode = [map(int, s.split('-')) for s in connected_mode]
        # map each layer index to its early connected layer index: {1: [0], 2: [1], 3: [0]}
        connected_mapping = {}
        for i, j in connected_mode:
            if j not in connected_mapping:
                connected_mapping[j] = [i]
            else:
                connected_mapping[j] = connected_mapping[j].append(i)

        net_collections = [input_layer]
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with tf.variable_scope('dnn_{}/hiddenlayer_{}'.format(model_id, layer_id),
                    values=(net,)) as hidden_layer_scope:
                net = tf.layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=ACTIVATION_FN,
                    kernel_initializer=tf.glorot_uniform_initializer(),  # also called Xavier uniform initializer.
                    kernel_regularizer=REG,
                    name=hidden_layer_scope)
                if DROPOUT is not None and mode == tf.estimator.ModeKeys.TRAIN:
                    net = tf.layers.dropout(net, rate=DROPOUT, training=True)
                if BATCH_NORM:
                    net = tf.layers.batch_normalization(net)
                connect_net_collections = [net for idx, net in enumerate(net_collections) if idx in connected_mapping[layer_id + 1]]
                connect_net_collections.append(net)
                net = tf.concat(connect_net_collections, axis=1)
                net_collections.append(net)
            add_layer_summary(net, hidden_layer_scope.name)

    with tf.variable_scope('dnn_{}/logits'.format(model_id), values=(net,)) as logits_scope:
        logits = tf.layers.dense(
                net,
                units=units,
                kernel_initializer=tf.glorot_uniform_initializer(),
                kernel_regularizer=REG,
                name=logits_scope)
    add_layer_summary(logits, logits_scope.name)
    return logits


def multidnn_logit_fn_builder(units, hidden_units_list,
                              connected_mode_list, feature_columns, input_layer_partitioner):
    """Multi dnn logit function builder.
    Args:
        hidden_units_list: 1D iterable list for single dnn or 2D for multi dnn.
            if use single format, default to use same hidden_units in all multi dnn.
            eg: [128, 64, 32] or [[128, 64, 32], [64, 32]]
        connected_mode_list: iterable list of {`simple`, `first_dense`, `last_dense`, `dense`, `resnet`} 
            consistent with above hidden_units_list. 
            if use single format, default to use same connected_mode in all multi dnn.
            eg: `simple` or [`simple`, `first_dense`] or [0-1, 0-3] or [[0-1, 0-3], [0-1]]
    Returns:
        multidnn logit fn.
    """
    if not isinstance(units, int):
        raise ValueError('units must be an int. Given type: {}'.format(type(units)))
    if not isinstance(hidden_units_list[0], (list, tuple)):
        hidden_units_list = [hidden_units_list]  # compatible for single dnn input hidden_units
        # raise ValueError('multi dnn hidden_units must be a 2D list or tuple. Given: {}'.format(hidden_units_list))
    if isinstance(connected_mode_list, str) or \
            (isinstance(connected_mode_list[0], str) and len(connected_mode_list[0]) == 3):  # `simple`
        connected_mode_list = [connected_mode_list] * len(hidden_units_list)

    def multidnn_logit_fn(features, mode):
        logits = []
        for idx, (hidden_units, connected_mode) in enumerate(zip(hidden_units_list, connected_mode_list)):
            logits.append(
                _dnn_logit_fn(
                    features,
                    mode,
                    idx + 1,
                    units,
                    hidden_units,
                    connected_mode,
                    feature_columns,
                    input_layer_partitioner))
        logits = tf.add_n(logits)  # Adds all input tensors element-wise.
        return logits
    return multidnn_logit_fn


class DNN:
    def __init__(self,
                 hidden_units,
                 connected_layers=None,
                 activation_fn=tf.nn.relu,
                 batch_norm=None,
                 dropout=None):
        """
        hidden_units: 
            Iterable of number hidden units per layer. All layers are
            fully connected. Ex. `[64, 32]` means first layer has 64 nodes and
            second one has 32.
        """
        self.hidden_units = hidden_units
        self.connected_layers = connected_layers
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.batch_norm = batch_norm


class MultiDNNClassifier(tf.estimator.Estimator):
    """
    A classifier for Multi DNN joint models based on tf.estimator.DNNClassifier.
    """
    def __init__(self,
               model_collections,
               feature_columns,
               model_dir=None,
               n_classes=2,
               weight_column=None,
               label_vocabulary=None,
               optimizer='Adagrad',
               input_layer_partitioner=None,
               config=None):
        """Initializes a `DNNClassifier` instance.
            Args:
               feature_columns: An iterable containing all the feature columns used by
                 the model. All items in the set should be instances of classes derived
                 from `_FeatureColumn`.
               model_dir: Directory to save model parameters, graph and etc. This can
                 also be used to load checkpoints from the directory into a estimator to
                 continue training a previously saved model.
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
               optimizer: An instance of `tf.Optimizer` used to train the model. Defaults
                 to Adagrad optimizer.
               activation_fn: Activation function applied to each layer. If `None`, will
                 use `tf.nn.relu`.
               dropout: When not `None`, the probability we will drop out a given
                 coordinate.
               input_layer_partitioner: Optional. Partitioner for input layer. Defaults
                 to `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
               config: `RunConfig` object to configure the runtime settings.
        """
        if not model_collections:
            raise ValueError('Empty model collections, must fill DNN model instance.')
        assert isinstance(model_collections, (list, tuple)), "model_collections must be a list or tuple"
        for model in model_collections:
            if not isinstance(model, DNN):
                raise ValueError("model_collections element must be an instance of class DNN")
        if n_classes == 2:
            head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(  # pylint: disable=protected-access
                weight_column=weight_column,
                label_vocabulary=label_vocabulary)
        else:
            head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(  # pylint: disable=protected-access
                n_classes, weight_column=weight_column,
                label_vocabulary=label_vocabulary)

        def _dnn_model_fn(
            features, labels, mode, head,
            optimizer='Adagrad', input_layer_partitioner=None, config=None):
            """Deep Neural Net model_fn.
            Args:
              features: dict of `Tensor`.
              labels: `Tensor` of shape [batch_size, 1] or [batch_size] labels of
                dtype `int32` or `int64` in the range `[0, n_classes)`.
              mode: Defines whether this is training, evaluation or prediction.
                See `ModeKeys`.
              head: A `head_lib._Head` instance.
              hidden_units: Iterable of integer number of hidden units per layer.
              feature_columns: Iterable of `feature_column._FeatureColumn` model inputs.
              optimizer: String, `tf.Optimizer` object, or callable that creates the
                optimizer to use for training. If not specified, will use the Adagrad
                optimizer with a default learning rate of 0.05.
              activation_fn: Activation function applied to each layer.
              dropout: When not `None`, the probability we will drop out a given
                coordinate.
              input_layer_partitioner: Partitioner for input layer. Defaults
                to `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
              config: `RunConfig` object to configure the runtime settings.
            Returns:
              predictions: A dict of `Tensor` objects.
              loss: A scalar containing the loss of the step.
              train_op: The op for training.
            Raises:
              ValueError: If features has the wrong type.
            """
            if not isinstance(features, dict):
                raise ValueError('features should be a dictionary of `Tensor`s. '
                             'Given type: {}'.format(type(features)))
            optimizer = get_optimizer_instance(
                optimizer, learning_rate=0.05)
            num_ps_replicas = config.num_ps_replicas if config else 0

            partitioner = tf.min_max_variable_partitioner(
                max_partitions=num_ps_replicas)
            with tf.variable_scope(
                'dnn',
                values=tuple(six.itervalues(features)),
                partitioner=partitioner):
                input_layer_partitioner = input_layer_partitioner or (
                    tf.min_max_variable_partitioner(
                        max_partitions=num_ps_replicas,
                        min_slice_size=64 << 20))
                # unit is num_classes, shape(batch_size, num_classes)
                logits = []
                for idx, m in enumerate(model_collections):
                    logits.append(
                        _dnn_logit_fn(
                            features,
                            mode,
                            idx+1,
                            head.logits_dimension,
                            m.hidden_units,
                            m.connected_layers,
                            feature_columns,
                            input_layer_partitioner))
                logits = tf.add_n(logits)  # add logit layer is same with concactenate the layer before logit layer

                def _train_op_fn(loss):
                    """Returns the op to optimize the loss."""
                    return optimizer.minimize(
                        loss, global_step=tf.train.get_global_step())
            return head.create_estimator_spec(
                features=features,
                mode=mode,
                labels=labels,
                train_op_fn=_train_op_fn,
                logits=logits)

        def _model_fn(features, labels, mode, config):
            return _dnn_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                head=head,
                optimizer=optimizer,
                input_layer_partitioner=input_layer_partitioner,
                config=config)
        super(MultiDNNClassifier, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config)


if __name__ == '__main__':
    dnn_1 = DNN(hidden_units=[128])
    dnn_2 = DNN(hidden_units=[128, 64])
    dnn_3 = DNN(hidden_units=[256, 128, 64])

    multi_dnn = MultiDNNClassifier(
        model_collections=(dnn_1, dnn_2, dnn_3),
        feature_columns="deep_columns")


