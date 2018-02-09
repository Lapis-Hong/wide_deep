#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/2/9
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf
from tensorflow.python.estimator.canned import head as head_lib

from lib import util

_LEARNING_RATE = 0.001  # 0.05


def _add_layer_summary(value, tag):
    tf.summary.scalar('%s/fraction_of_zero_values' % tag, tf.nn.zero_fraction(value))
    tf.summary.histogram('%s/activation' % tag, value)


def _dnn_logit_fn_builder(model_id, units, hidden_units, connected_layers, feature_columns,
                          activation_fn, dropout, batch_norm, input_layer_partitioner):
    """Function builder for a dnn logit_fn.
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
   """
    if not isinstance(units, int):
        raise ValueError('units must be an int.  Given type: {}'.format(type(units)))

    def dnn_logit_fn(features, mode):
        """Deep Neural Network logit_fn.
        Args:
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
                '{}/input_from_feature_columns'.format(model_id),
                values=tuple(six.itervalues(features)),
                partitioner=input_layer_partitioner):
            net = tf.feature_column.input_layer(features=features,
                                                feature_columns=feature_columns)
        input_layer = net
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with tf.variable_scope('{}/hiddenlayer_{}d'.format(model_id, layer_id),
                                    values=(net,)) as hidden_layer_scope:
                net = tf.layers.dense(
                        net,
                        units=num_hidden_units,
                        activation=activation_fn,
                        kernel_initializer=tf.glorot_uniform_initializer(),  # also called Xavier uniform initializer.
                        name=hidden_layer_scope)
                if connected_layers:
                    # tf.logging.info('Using connected_layers')
                    # print(input_layer.get_shape()) (?, 784)
                    # print(net.get_shape()) (?, 784
                    net = tf.concat([net, input_layer], axis=1)  # TODO: arbitrary connections
                if dropout is not None and mode == tf.estimator.ModeKeys.TRAIN:
                    net = tf.layers.dropout(net, rate=dropout, training=True)  # rate=0.1 would drop out 10% of input units.
                if batch_norm:
                    net = tf.layers.batch_normalization(net)
            _add_layer_summary(net, hidden_layer_scope.name)

        with tf.variable_scope('{}/logits'.format(model_id), values=(net,)) as logits_scope:
            logits = tf.layers.dense(
                    net,
                    units=units,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    name=logits_scope)
        _add_layer_summary(logits, logits_scope.name)
        return logits

    return dnn_logit_fn


class DNN:
    def __init__(self,
                 hidden_units,
                 connected_layers=None,
                 activation_fn=tf.nn.relu,
                 batch_norm=None,
                 dropout=None,
                 ):
        self.hidden_units = hidden_units
        self.connected_layers = connected_layers
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.batch_norm = batch_norm


class MultiDNNClassifier(tf.estimator.Estimator):
  """A classifier for TensorFlow DNN models.
  Example:
  ```python
  categorical_feature_a = categorical_column_with_hash_bucket(...)
  categorical_feature_b = categorical_column_with_hash_bucket(...)
  categorical_feature_a_emb = embedding_column(
      categorical_column=categorical_feature_a, ...)
  categorical_feature_b_emb = embedding_column(
      categorical_column=categorical_feature_b, ...)
  estimator = DNNClassifier(
      feature_columns=[categorical_feature_a_emb, categorical_feature_b_emb],
      hidden_units=[1024, 512, 256])
  # Or estimator using the ProximalAdagradOptimizer optimizer with
  # regularization.
  estimator = DNNClassifier(
      feature_columns=[categorical_feature_a_emb, categorical_feature_b_emb],
      hidden_units=[1024, 512, 256],
      optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
      ))
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
  * if `weight_column` is not `None`, a feature with
    `key=weight_column` whose value is a `Tensor`.
  * for each `column` in `feature_columns`:
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
               model_collections,
               feature_columns,
               model_dir=None,
               n_classes=2,
               weight_column=None,
               label_vocabulary=None,
               optimizer='Adagrad',
               input_layer_partitioner=None,
               config=None):
    if not model_collections:
        raise ValueError('Empty model collections, must fill DNN model instance.')
    assert isinstance(model_collections, (list, tuple)), "model_collections must be a list or tuple"
    for model in model_collections:
        if not isinstance(model, DNN):
            raise ValueError("model_collections element must be an instance of class DNN")

    """Initializes a `DNNClassifier` instance.
    Args:
      hidden_units: Iterable of number hidden units per layer. All layers are
        fully connected. Ex. `[64, 32]` means first layer has 64 nodes and
        second one has 32.
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
        optimizer = util.get_optimizer_instance(
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
                logits.append(_dnn_logit_fn_builder(idx+1,
                                                head.logits_dimension,
                                                m.hidden_units,
                                                m.connected_layers,
                                                feature_columns,
                                                m.activation_fn,
                                                m.dropout,
                                                m.batch_norm,
                                                input_layer_partitioner)(features, mode))

            logits = tf.add_n(logits)  # add logit layer is same with concactenate the layer before logit layer

            def _train_op_fn(loss):
                """Returns the op to optimize the loss."""
                return optimizer.minimize(
                    loss,
                    global_step=tf.train.get_global_step())

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


