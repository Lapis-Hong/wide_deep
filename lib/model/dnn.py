#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/2/9
"""This module based on tf.estimator.DNNClassifier.
Dnn logits builder. 
Extend dnn architecture, add BN layer, add skip connections similar to Resnet in Cnn.
Extend dnn to multi joint dnn.
"""
# TODO: self-made head
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf
from tensorflow.python.estimator.canned import head as head_lib

from lib.model.util import add_layer_summary, get_optimizer_instance

_LEARNING_RATE = 0.001  # 0.05


# def _dnn_logit_fn_builder(model_id, units, hidden_units, connected_mode, feature_columns,
#                           activation_fn, dropout, batch_norm, input_layer_partitioner):
    # if not isinstance(units, int):
    #     raise ValueError('units must be an int.  Given type: {}'.format(type(units)))

def _dnn_logit_fn(features, mode, model_id, units, hidden_units, connected_mode, feature_columns,
                          activation_fn, dropout, batch_norm, input_layer_partitioner):
    """Deep Neural Network logit_fn.
    Args:
        Function builder for a dnn logit_fn.
    Args:
      units: An int indicating the dimension of the logit layer.  In the
        MultiHead case, this should be the sum of all component Heads' logit
        dimensions.
      hidden_units: Iterable of integer number of hidden units per layer.
      feature_columns: Iterable of `feature_column._FeatureColumn` model inputs.
      activation_fn: Activation function applied to each layer.
      dropout: When not `None`, the probability we will drop out a given coordinate.
      batch_norm: Bool, Whether to use BN in dnn.
      input_layer_partitioner: Partitioner for input layer.
    Returns:
      A logit_fn (see below).
    Raises:
      ValueError: If units is not an int.

      features: This is the first item returned from the `input_fn`
                passed to `train`, `evaluate`, and `predict`. This should be a
                single `Tensor` or `dict` of same.
      mode: Optional. Specifies if this training, evaluation or prediction. See
            `ModeKeys`.
    Returns:
      A `Tensor` representing the logits, or a list of `Tensor`'s representing
      multiple logits in the MultiHead case.
    """
    with tf.variable_scope(
            'input_from_feature_columns',
            values=tuple(six.itervalues(features)),
            partitioner=input_layer_partitioner):
        net = tf.feature_column.input_layer(
            features=features,
            feature_columns=feature_columns)
    input_layer = net
    if connected_mode == 'simple':
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with tf.variable_scope('dnn_{}/hiddenlayer_{}d'.format(model_id, layer_id),
                    values=(net,)) as hidden_layer_scope:
                net = tf.layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=activation_fn,
                    kernel_initializer=tf.glorot_uniform_initializer(),  # also called Xavier uniform initializer.
                    name=hidden_layer_scope)
                if dropout is not None and mode == tf.estimator.ModeKeys.TRAIN:
                    net = tf.layers.dropout(net, rate=dropout, training=True)  # rate=0.1 would drop out 10% of input units.
                if batch_norm:
                    net = tf.layers.batch_normalization(net)
            add_layer_summary(net, hidden_layer_scope.name)

    elif connected_mode == 'first_dense':
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with tf.variable_scope('dnn_{}/hiddenlayer_{}d'.format(model_id, layer_id),
                    values=(net,)) as hidden_layer_scope:
                net = tf.layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=activation_fn,
                    kernel_initializer=tf.glorot_uniform_initializer(),  # also called Xavier uniform initializer.
                    name=hidden_layer_scope)
                if dropout is not None and mode == tf.estimator.ModeKeys.TRAIN:
                    net = tf.layers.dropout(net, rate=dropout, training=True)  # rate=0.1 would drop out 10% of input units.
                if batch_norm:
                    net = tf.layers.batch_normalization(net)
                net = tf.concat([net, input_layer], axis=1)
            add_layer_summary(net, hidden_layer_scope.name)

    elif connected_mode == 'last_dense':
        net_collections = [input_layer]
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with tf.variable_scope('dnn_{}/hiddenlayer_{}d'.format(model_id, layer_id),
                    values=(net,)) as hidden_layer_scope:
                net = tf.layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=activation_fn,
                    kernel_initializer=tf.glorot_uniform_initializer(),  # also called Xavier uniform initializer.
                    name=hidden_layer_scope)
                if dropout is not None and mode == tf.estimator.ModeKeys.TRAIN:
                    net = tf.layers.dropout(net, rate=dropout, training=True)  # rate=0.1 would drop out 10% of input units.
                if batch_norm:
                    net = tf.layers.batch_normalization(net)
                net_collections.append(net)
            add_layer_summary(net, hidden_layer_scope.name)
        net = tf.concat(net_collections, axis=1)  # Concatenates the list of tensors `values` along dimension `axis`

    elif connected_mode == 'dense':
        net_collections = [input_layer]
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with tf.variable_scope('dnn_{}/hiddenlayer_{}d'.format(model_id, layer_id),
                    values=(net,)) as hidden_layer_scope:
                net = tf.layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=activation_fn,
                    kernel_initializer=tf.glorot_uniform_initializer(),  # also called Xavier uniform initializer.
                    name=hidden_layer_scope)
                if dropout is not None and mode == tf.estimator.ModeKeys.TRAIN:
                    net = tf.layers.dropout(net, rate=dropout, training=True)  # rate=0.1 would drop out 10% of input units.
                if batch_norm:
                    net = tf.layers.batch_normalization(net)
                net_collections.append(net)
                net = tf.concat(net_collections, axis=1)
            add_layer_summary(net, hidden_layer_scope.name)

    elif connected_mode == 'resnet':  # connect layers in turn 0-1; 1-2; 2-3;
        net_collections = [input_layer]
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with tf.variable_scope('dnn_{}/hiddenlayer_{}d'.format(model_id, layer_id),
                    values=(net,)) as hidden_layer_scope:
                net = tf.layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=activation_fn,
                    kernel_initializer=tf.glorot_uniform_initializer(),  # also called Xavier uniform initializer.
                    name=hidden_layer_scope)
                if dropout is not None and mode == tf.estimator.ModeKeys.TRAIN:
                    net = tf.layers.dropout(net, rate=dropout, training=True)  # rate=0.1 would drop out 10% of input units.
                if batch_norm:
                    net = tf.layers.batch_normalization(net)
                net = tf.concat([net, net_collections[layer_id + 1 - 1]], axis=1)
                net_collections.append(net)
            add_layer_summary(net, hidden_layer_scope.name)

    else:  # arbitrary connections,  pairs like (0, 1), (0, 3), (1, 3), small index layer first
        connected_mapping = {}  # map each layer index to its early connected layer index
        for i, j in connected_mode:
            if i not in connected_mapping:
                connected_mapping[j] = [i]
            else:
                connected_mapping[j] = connected_mapping[j].append(i)

        net_collections = [input_layer]
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with tf.variable_scope('dnn_{}/hiddenlayer_{}d'.format(model_id, layer_id),
                    values=(net,)) as hidden_layer_scope:
                net = tf.layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=activation_fn,
                    kernel_initializer=tf.glorot_uniform_initializer(),  # also called Xavier uniform initializer.
                    name=hidden_layer_scope)
                if dropout is not None and mode == tf.estimator.ModeKeys.TRAIN:
                    net = tf.layers.dropout(net, rate=dropout, training=True)  # rate=0.1 would drop out 10% of input units.
                if batch_norm:
                    net = tf.layers.batch_normalization(net)
                connect_net_collections = [net for idx, net in enumerate(net_collections) if idx in connected_mapping[layer_id + 1]]
                net = tf.concat(connect_net_collections.append(net), axis=1)
                net_collections.append(net)
            add_layer_summary(net, hidden_layer_scope.name)

    with tf.variable_scope('{}/logits'.format(model_id), values=(net,)) as logits_scope:
        logits = tf.layers.dense(
                net,
                units=units,
                kernel_initializer=tf.glorot_uniform_initializer(),
                name=logits_scope)
    add_layer_summary(logits, logits_scope.name)
    return logits

    # return dnn_logit_fn


def _multidnn_logit_fn_builder(units, hidden_units_list, connected_mode_list, feature_columns,
                          activation_fn, dropout, batch_norm, input_layer_partitioner):
    if not isinstance(units, int):
        raise ValueError('units must be an int. Given type: {}'.format(type(units)))
    if not isinstance(hidden_units_list[0], (list, tuple)):
        hidden_units_list = [hidden_units_list]  # compatible for single dnn input hidden_units
        # raise ValueError('multi dnn hidden_units must be a 2D list or tuple. Given: {}'.format(hidden_units_list))
    if not isinstance(connected_mode_list, (list, tuple)):
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
                    activation_fn,
                    dropout,
                    batch_norm,
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
                optimizer, learning_rate=_LEARNING_RATE)
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
                            m.activation_fn,
                            m.dropout,
                            m.batch_norm,
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


