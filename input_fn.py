#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/24
from collections import OrderedDict
import tensorflow as tf

from read_conf import Config


CONFIG = Config()

def _column_to_csv_defaults():
    """parse columns to record_defaults param in tf.decode_csv func
    Return: _CSV_COLUMN_DEFAULTS Ordereddict {'feature name': [''],...}
    """
    _CSV_COLUMN_DEFAULTS = OrderedDict()
    _CSV_COLUMN_DEFAULTS['label'] = [0]  # first label default, empty if the field is must
    feature_all = Config.get_feature_name()
    feature_conf_dic = Config.read_feature_conf()
    for f in feature_all:
        if f in feature_conf_dic:  # used features
            conf = feature_conf_dic[f]
            if conf['feature_type'] == 'category':
                if conf['feature_transform'] == 'identity':  # identity category column need int type
                    _CSV_COLUMN_DEFAULTS[f] = [0]
                else:
                    _CSV_COLUMN_DEFAULTS[f] = ['']
            else:
                _CSV_COLUMN_DEFAULTS[f] = [0.0]  # 0.0 for float32
        else:  # unused features
            _CSV_COLUMN_DEFAULTS[f] = ['']
    return _CSV_COLUMN_DEFAULTS


def _column_to_dtype():
    """Parse columns to tf.dtype
     Return: similar to _csv_column_defaults()
     """
    _column_dtype_dic = OrderedDict()
    _column_dtype_dic['label'] = tf.int32
    feature_all = Config.get_feature_name()
    feature_conf_dic = Config.read_feature_conf()
    for f in feature_all:
        if f in feature_conf_dic:
            conf = feature_conf_dic[f]
            if conf['feature_type'] == 'category':
                if conf['feature_transform'] == 'identity':  # identity category column need int type
                    _column_dtype_dic[f] = tf.int32
                else:
                    _column_dtype_dic[f] = tf.string
            else:
                _column_dtype_dic[f] = tf.float32  # 0.0 for float32
        else:
            _column_dtype_dic[f] = tf.string
    return _column_dtype_dic


def _parse_csv(value):  # value: Tensor("arg0:0", shape=(), dtype=string)
    _CSV_COLUMN_DEFAULTS = _column_to_csv_defaults()
    feature_unused = set(Config.get_feature_name('all'))-set(Config.get_feature_name('used'))
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS.values(), field_delim='\t', use_quote_delim=True, na_value='-')
    # na_value fill with record_defaults
    # `tf.decode_csv` return Tensor list: <tf.Tensor 'DecodeCSV:60' shape=() dtype=string>  rank 0 Tensor
    # columns = (tf.expand_dims(col, 0) for col in columns)  # fix rank 0 error for dataset.padded_patch()
    features = dict(zip(_CSV_COLUMN_DEFAULTS.keys(), columns))
    for f in feature_unused:
        features.pop(f)  # remove unused features
    labels = features.pop('label')
    return features, tf.equal(labels, 1)


def _parse_csv_multivalue(value):
    _CSV_COLUMN_DEFAULTS = _column_to_csv_defaults()
    feature_unused = set(Config.get_feature_name('all')) - set(Config.get_feature_name('used'))
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS.values(), field_delim='\t', na_value='-')
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
    feature_unused = set(Config.get_feature_name('all')) - set(Config.get_feature_name('used'))
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


def input_fn(data_file, num_epochs, batch_size, shuffle=True, multivalue=False):
    """
    Generate an input function for the Estimator.
    data_file: can be both file or directory
    parse_func: custom data file parser function
    """
    # filename_queue = tf.train.string_input_producer([
    #     "hdfs://namenode:8020/path/to/file1.csv",
    #     "hdfs://namenode:8020/path/to/file2.csv",
    # ])
    # assert tf.gfile.Exists(data_file), (
    #   '%s not found. Please make sure you have either default data_file or '
    #   'set both arguments --train_data and --test_data.' % data_file)
    # if tf.gfile.IsDirectory(data_file):
    #     data_file_list = [f for f in tf.gfile.ListDirectory(data_file) if not f.startswith('.')]
    #     data_file = [data_file+'/'+file_name for file_name in data_file_list]

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=CONFIG.shuffle_buffer_size, seed=123)  # set of Tensor object
    tf.logging.info('Parsing input files: {}'.format(data_file))
    # Use `Dataset.map()` to build a pair of a feature dictionary
    # and a label tensor for each example.
    if multivalue:
        dataset = dataset.map(_parse_csv_multivalue, num_parallel_calls=24)
        # We call repeat after shuffling, rather than before, to prevent separate
        # epochs from blending together.
        dataset = dataset.repeat(num_epochs)
        padding_dic = {k: [None] for k in Config.get_feature_name('used')}
        dataset.padded_batch(batch_size, padded_shapes=(padding_dic, [1]))  # rank no change
    else:
        dataset = dataset.map(_parse_csv, num_parallel_calls=24)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)  # each element tensor must have exactly same shape, change rank 0 to rank 1

    iterator = dataset.make_one_shot_iterator()
    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    features, labels = iterator.get_next()
    return features, labels


def _parse_func_test(batch_size=5, multivalue=False):
    print('batch size: {}; multivalue: {}'.format(batch_size, multivalue))
    sess = tf.InteractiveSession()
    features, labels = input_fn('./data/train', 1, batch_size=batch_size, multivalue=multivalue)
    print(features)
    print(labels)
    for f, v in features.items()[:5]:
        print('{}: {}'.format(f, sess.run(v)))
    print('labels: {}'.format(sess.run(labels)))


def _input_tensor_test(batch_size=5, multivalue=False):
    sess = tf.InteractiveSession()
    features, labels = input_fn('./data/train', 1, batch_size=batch_size, multivalue=multivalue)
    print(features['ucomp'].eval())
    print(features['city_id'].eval())
    # categorical_column* can handle multivalue feature as a multihot
    # test for categorical_column and cross_column
    ucomp = tf.feature_column.categorical_column_with_hash_bucket('ucomp', 10)
    city_id = tf.feature_column.categorical_column_with_hash_bucket('city_id', 10)
    ucomp_X_city_id = tf.feature_column.crossed_column(['ucomp', 'city_id'], 10)
    for f in [ucomp, city_id, ucomp_X_city_id]:
        f_dense = tf.feature_column.indicator_column(f)
        # f_embed = tf.feature_column.embedding_column(f, 5)
        # sess.run(tf.global_variables_initializer())
        # input_tensor = tf.feature_column.input_layer(features, f_embed)
        input_tensor = tf.feature_column.input_layer(features, f_dense)
        print('{} input tensor:\n {}'.format(f, input_tensor.eval()))
    # dense_tensor = tf.feature_column.input_layer(features, [ucomp, city_id, ucomp_X_city_id])
    # print('total input tensor:\n {}'.format(sess.run(dense_tensor)))

    # wide_columns, deep_columns = build_model_columns()
    # dense_tensor = tf.feature_column.input_layer(features, deep_columns)
    # sess.run(tf.global_variables_initializer())  # fix Attempting to use uninitialized value error.
    # sess.run(tf.tables_initializer())  # fix Table not initialized error.
    # print(sess.run(dense_tensor))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    _parse_func_test(multivalue=False)
    _parse_func_test(multivalue=True)
    _parse_func_test(1, multivalue=True)

    # _input_tensor_test(multivalue=False)
    # _input_tensor_test(multivalue=True)

