#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/15
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from feature import build_model_columns, get_feature_name, _column_to_dtype, _csv_column_defaults


def build_estimator(model_dir, model_type):
    """Build an estimator appropriate for the given model type."""
    wide_columns, deep_columns = build_model_columns()
    hidden_units = [1024, 512, 256]  # [100, 75, 50, 25]
    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))

    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns,
            weight_column=None,
            optimizer=tf.train.FtrlOptimizer(
                learning_rate=0.1,
                l1_regularization_strength=0.5,
                l2_regularization_strength=1.0),  # 'Ftrl',
            partitioner=None,
            config=run_config)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            optimizer=tf.train.ProximalAdagradOptimizer(
                learning_rate=0.1,
                l1_regularization_strength=0.001,
                l2_regularization_strength=0.001),  # {'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'}
            activation_fn=tf.nn.relu,
            dropout=None,
            weight_column=None,
            input_layer_partitioner=None,
            config=run_config)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            linear_optimizer=tf.train.FtrlOptimizer(
                learning_rate=0.1,
                l1_regularization_strength=0.001,
                l2_regularization_strength=0.001),
            dnn_feature_columns=deep_columns,
            dnn_optimizer=tf.train.ProximalAdagradOptimizer(
                learning_rate=0.1,
                l1_regularization_strength=0.001,
                l2_regularization_strength=0.001),
            dnn_hidden_units=hidden_units,
            dnn_activation_fn=tf.nn.relu,
            dnn_dropout=None,
            n_classes=2,
            weight_column=None,
            label_vocabulary=None,
            input_layer_partitioner=None,
            config=run_config)


def input_fn(data_file, parse_func, num_epochs, batch_size, shuffle=True):
    """
    Generate an input function for the Estimator.
    data_file: can be both file or directory
    parse_func: custom data file parser function
    """
    # filename_queue = tf.train.string_input_producer([
    #     "hdfs://namenode:8020/path/to/file1.csv",
    #     "hdfs://namenode:8020/path/to/file2.csv",
    # ])
    assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have either default data_file or '
      'set both arguments --train_data and --test_data.' % data_file)
    if tf.gfile.IsDirectory(data_file):
        data_file_list = [f for f in tf.gfile.ListDirectory(data_file) if not f.startswith('.')]
        data_file = [data_file+'/'+file_name for file_name in data_file_list]

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)  # set of Tensor object
    tf.logging.info('Parsing input files: {}'.format(data_file))
    # Use `Dataset.map()` to build a pair of a feature dictionary
    # and a label tensor for each example.
    dataset = dataset.map(parse_func, num_parallel_calls=5)
    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    padding_dic = {k: [None] for k in get_feature_name('used')}
    dataset.padded_batch(batch_size, padded_shapes=(padding_dic, [None]))
    #dataset = dataset.batch(batch_size)  # each element tensor must have exactly same shape

    iterator = dataset.make_one_shot_iterator()
    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    features, labels = iterator.get_next()
    return features, labels


def _parse_csv(value):  # value: Tensor("arg0:0", shape=(), dtype=string)
    _CSV_COLUMN_DEFAULTS = _csv_column_defaults()
    feature_unused = set(get_feature_name('all'))-set(get_feature_name('used'))
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS.values(), field_delim='\t', na_value='-')
    # print(columns)
    # na_value fill with record_defaults
    # `tf.decode_csv` return Tensor list: <tf.Tensor 'DecodeCSV:60' shape=() dtype=string>
    features = dict(zip(_CSV_COLUMN_DEFAULTS.keys(), columns))
    for f in feature_unused:
        features.pop(f)  # remove unused features
    labels = features.pop('label')
    return features, tf.equal(labels, 1)


def _parse_csv2(value):
    _CSV_COLUMN_DEFAULTS = _csv_column_defaults()
    feature_unused = set(get_feature_name('all')) - set(get_feature_name('used'))
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS.values(), field_delim='\t', na_value='-')
    # cloumns = [tf.string_split(col, ",") for col in columns]
    # `tf.decode_csv` return Tensor list: <tf.Tensor 'DecodeCSV:60' shape=() dtype=string>
    features = {}
    for f, tensor in zip(_CSV_COLUMN_DEFAULTS.keys(), columns):
        if f in feature_unused:
            continue
        if isinstance(_CSV_COLUMN_DEFAULTS[f][0], str):
            features[f] = tf.string_split([tensor], ",").values  # tensor shape (?,)
        else:
            # features[f] = tensor  # error
            features[f] = tf.expand_dims(tensor, 0)  # change shape from () to (1,)
    labels = features.pop('label')
    return features, tf.equal(labels, 1)


def _parse_line(value):
    """Parse input Tensor to Tensors"""
    # TODO: deal with missing value '-'
    column_dtype_dic = _column_to_dtype()
    feature_unused = set(get_feature_name('all')) - set(get_feature_name('used'))
    col_names = column_dtype_dic.keys()
    n_feature = len(col_names)
    st = tf.string_split([value], '\t', skip_empty=True)  # input must be rank 1, return SparseTensor
    # print(st.values)  # <tf.Tensor 'StringSplit_11:1' shape=(?,) dtype=string>
    col_values = [st.values[i] for i in range(n_feature)]  # <tf.Tensor 'strided_slice_2:0' shape=() dtype=string>
    features = dict(zip(col_names, col_values))
    # label, feature = st.values[0], st.values[1:]  # array(['a', 'b', 'c', 'd'], dtype=object
    label_name = col_names[0]
    labels = tf.string_to_number(features.pop(label_name), tf.int32, name=label_name)
    for f_name, f_value in features.items():
        dtype = column_dtype_dic[f_name]
        if dtype == tf.string:
            tensor = tf.string_split([f_value], ',', skip_empty=True).values
            # print(tensor.shape)  # (?,)
            features[f_name] = tf.string_split([f_value], ',', skip_empty=True).values
        else:
            features[f_name] = f_value
            # print(f_value.shape)
            features[f_name] = tf.string_to_number(f_value, out_type=dtype, name=f_name)  # cast, to_int32, to_float do not suppot string to int or float
    return features, labels


def _parse_func_test(func_name):
    sess = tf.InteractiveSession()
    dataset = tf.data.TextLineDataset('./data/train/train1')
    dataset = dataset.map(func_name, num_parallel_calls=5)
    #dataset = dataset.batch(5)
    padding_shape = {k: [None] for k in get_feature_name('used')}
    # padding_value = {k: 'padding' for k in get_feature_name('used')}
    dataset = dataset.padded_batch(5, padded_shapes=(padding_shape, [None]))  # pad each feature shape None
    iterator = dataset.make_one_shot_iterator()
    for _ in range(1):
        features, labels = iterator.get_next()
        print('features', sess.run(features))
        print('label', sess.run(labels))
        for k, v in features.items():
            print('{}:\n{}'.format(k, sess.run(v)))
        print('-' * 50)
        print(features['ucomp'].eval())
        print(features['city_id'].eval())
        # categorical_column* can handle multivalue feature as a multihot
        # test for categorical_column and cross_column
        ucomp = tf.feature_column.categorical_column_with_hash_bucket('ucomp', 10)
        city_id = tf.feature_column.categorical_column_with_hash_bucket('city_id', 10)
        ucomp_X_city_id = tf.feature_column.crossed_column(['ucomp', 'city_id'], 10)
        for f in [ucomp, city_id, ucomp_X_city_id]:
            f_dense = tf.feature_column.indicator_column(f)
            input_tensor = tf.feature_column.input_layer(features, f_dense)
            print('{} input tensor:\n {}'.format(f, input_tensor.eval()))
        # dense_tensor = tf.feature_column.input_layer(features, [city_id, ctr_strategy_type, ctr_stragety_type_X_city_id])
        # print(sess.run(dense_tensor))

        # wide_columns, deep_columns = build_model_columns()
        # dense_tensor = tf.feature_column.input_layer(features, deep_columns)
        # sess.run(tf.global_variables_initializer())  # fix Attempting to use uninitialized value error.
        # sess.run(tf.tables_initializer())  # fix Table not initialized error.
        # print(sess.run(dense_tensor))


if __name__ == '__main__':
    _parse_func_test(_parse_csv2)
    model = build_estimator('./model', 'wide')
    # print(model.config)  # <tensorflow.python.estimator.run_config.RunConfig object at 0x118de4e10>
    # print(model.model_dir)  # ./model
    # print(model.model_fn)  # <function public_model_fn at 0x118de7b18>
    # print(model.params)  # {}
    # print(model.get_variable_names())
    # print(model.get_variable_value('dnn/hiddenlayer_0/bias'))
    # print(model.get_variable_value('dnn/hiddenlayer_0/bias/Adagrad'))
    # print(model.get_variable_value('dnn/hiddenlayer_0/kernel'))
    # print(model.latest_checkpoint())  # another 4 method is export_savedmodel,train evaluate predict