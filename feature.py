#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/12
from collections import OrderedDict

import numpy as np
import tensorflow as tf

# wide columns
categorical_column_with_identity = tf.feature_column.categorical_column_with_identity
categorical_column_with_hash_bucket = tf.feature_column.categorical_column_with_hash_bucket
categorical_column_with_vocabulary_list = tf.feature_column.categorical_column_with_vocabulary_list
crossed_column = tf.feature_column.crossed_column
bucketized_column = tf.feature_column.bucketized_column
# deep columns
embedding_column = tf.feature_column.embedding_column
indicator_column = tf.feature_column.indicator_column
numeric_column = tf.feature_column.numeric_column


def _read_data_schema():
    """Read data schema
    Return: 
        a list, raw data fields names in order, including the label
    """
    for line in open('./conf/data.schema'):
        line = line.strip().strip('\n')
        if line.startswith('#') or not line:
            continue
        return [w.strip().lower() for w in line.split(',')]


def _read_feature_conf():
    """Read feature configuration, parse all used features conf
    Usage: 
    feature_conf_dic = read_feature_conf()
    f_param = feature_conf_dic['request_id'].get('feature_parammeter'))
    Return: 
        {'request_id', {'feature_type': 'category', 'feature_transform': 'hash_bucket', 'feature_parameter': [500000]}, ...}"""
    feature_schema = _read_data_schema()[1:]
    feature_conf_dic = OrderedDict()
    for nu, line in enumerate(open('./conf/feature.conf')):
        line = line.strip().strip('\n')
        if line.startswith('#') or not line:
            continue
        try:
            f_name, f_type, f_transform, f_param_str = [w.strip().lower() for w in line.split(';')]
        except ValueError:
            raise ValueError('Can not parse line {}, please check the format in feature.conf'.format(nu+1))
        assert f_name in feature_schema, 'Inconsistent feature name at line {}, see data.schema'.format(nu+1)
        assert f_type in {'category', 'continuous'}, 'Invalid feature type at line {}'.format(nu+1)
        assert f_transform in {'hash_bucket', 'vocab', 'identity', 'discretize', 'numeric'}, 'Invalid feature transform at line {}'.format(nu+1)
        f_param = f_param_str.split(',')  # ['0', '1'] string list
        feature_conf = {'feature_type': f_type, 'feature_transform': f_transform, 'feature_parameter': f_param}
        feature_conf_dic[f_name] = feature_conf
    return feature_conf_dic


def _read_cross_feature_conf():
    """Read cross feature configuration
    Return: list with tuple element, [(['adplan_id', 'category'], 10000, 1), (),...]
    """
    feature_used = _read_feature_conf().keys()
    cross_feature_list = []
    for nu, line in enumerate(open('./conf/cross_feature.conf')):
        line = line.strip().strip('\n')
        if line.startswith('#') or not line:
            continue
        try:
            cross_feature, hash_bucket_size, is_deep = [w.strip().lower() for w in line.split(';')]
        except ValueError:
            raise ValueError('Can not parse line {}, please check the format in cross_feature.conf'.format(nu+1))
            # print('Can not parse line {}, please check the format in cross_feature.conf'.format(nu+1))
            # exit()
            # exit('Can not parse line {}, please check the format in cross_feature.conf'.format(nu+1))
        cross_feature = [w.strip().lower() for w in cross_feature.split(',')]
        for f_name in cross_feature:  # check all cross feature is used
            assert f_name in feature_used, 'Invalid cross feature name at line {}, not found in feature.conf'.format(nu+1)
        assert len(cross_feature) > 1, 'Invalid at line {}, cross feature name at least 2'.format(nu+1)
        # hash_bucket_size = 1000*int(hash_bucket_size.replace('k', '')) if hash_bucket_size else 10000  # default 10k
        hash_bucket_size = 1000 * int(hash_bucket_size.replace('k', '') or 10)  # default 10k
        # is_deep = int(is_deep) if is_deep else 1  # default 1
        is_deep = int(is_deep or 1)  # default 1
        cross_feature_list.append((cross_feature, hash_bucket_size, is_deep))
    return cross_feature_list


def get_feature_name(feature_type='all'):
    """
    Args:
     feature_type: one of {'all', 'used', 'category', 'continuous'}
    Return: feature name list
    """
    feature_conf_dic = _read_feature_conf()
    if feature_type == 'all':
        return _read_data_schema()[1:]
    elif feature_type == 'used':
        return feature_conf_dic.keys()
    elif feature_type == 'category':
        return [feature for feature, conf in feature_conf_dic.items() if conf['feature_type'] == 'category']
    elif feature_type == 'continuous':
        return [feature for feature, conf in feature_conf_dic.items() if conf['feature_type'] == 'continuous']
    else:
        raise TypeError("Invalid parameter, must be one of 'all', 'used', 'category, 'continuous")


def build_model_columns():
    """
    Build wide and deep feature columns from custom feature conf using tf.feature_column API
    wide_columns: category features + cross_features + [discretized continuous features]
    deep_columns: continuous features + category features(onehot or embedding for sparse features) + [cross_features(embedding)]
    Return: _CategoricalColumn and __DenseColumn for tf.estimators API
    """
    def embedding_dim(dim):
        """empirical embedding dim"""
        return int(np.power(2, np.ceil(np.log(dim**0.25))))
    feature_conf_dic = _read_feature_conf()
    tf.logging.info('Total used feature class: {}'.format(len(feature_conf_dic)))
    cross_feature_list = _read_cross_feature_conf()
    tf.logging.info('Total used cross feature class: {}'.format(len(cross_feature_list)))
    wide_columns = []
    deep_columns = []
    wide_dim = 0
    deep_dim = 0
    for feature, conf in feature_conf_dic.items():
        f_type, f_tran, f_param = conf.values()
        if f_type == 'category':
            if f_tran == 'hash_bucket':
                hash_bucket_size = int(f_param[0])
                assert len(f_param) == 1, 'Invalid param conf for feature {}'.format(feature)
                col = categorical_column_with_hash_bucket(feature,
                                                          hash_bucket_size=hash_bucket_size,
                                                          dtype=tf.string)
                wide_columns.append(col)
                wide_dim += hash_bucket_size
                embed_dim = embedding_dim(hash_bucket_size)
                deep_columns.append(embedding_column(col,
                                                     dimension=embed_dim,
                                                     combiner='mean',
                                                     initializer=None,
                                                     ckpt_to_load_from=None,
                                                     tensor_name_in_ckpt=None,
                                                     max_norm=None,
                                                     trainable=True))
                deep_dim += embed_dim
            elif f_tran == 'vocab':
                col = categorical_column_with_vocabulary_list(feature,
                                                              vocabulary_list=f_param,
                                                              dtype=None,
                                                              default_value=-1,
                                                              num_oov_buckets=0)  # len(vocab)+num_oov_buckets
                wide_columns.append(col)
                wide_dim += len(f_param)
                deep_columns.append(indicator_column(col))
                deep_dim += len(f_param)
            elif f_tran == 'identity':
                num_buckets = int(f_param[0])
                col = categorical_column_with_identity(feature,
                                                       num_buckets=num_buckets,
                                                       default_value=0)  # Values outside range will result in default_value if specified, otherwise it will fail.
                wide_columns.append(col)
                wide_dim += num_buckets
                deep_columns.append(indicator_column(col))
                deep_dim += num_buckets
            else:
                raise TypeError('Invalid feature transform for feature {}'.format(feature))
        else:
            assert f_tran in {'numeric', 'discretize'}
            col = numeric_column(feature,
                                 shape=(1,),
                                 default_value=None,
                                 dtype=tf.float32,
                                 normalizer_fn=None)  # TODO，standard normalization
            if f_tran == 'discretize':  # whether include continuous features in wide part
                wide_columns.append(bucketized_column(col, boundaries=map(int, f_param)))
                wide_dim += (len(f_param)+1)
            deep_columns.append(col)
            deep_dim += 1

    for cross_features, hash_bucket_size, is_deep in cross_feature_list:
        cf_list = []
        for f in cross_features:
            f_type, f_tran, f_param = feature_conf_dic[f].values()
            if f_type == 'continuous':
                cf_list.append(bucketized_column(numeric_column(f), boundaries=map(int, f_param)))
            else:  # category col only put the name in crossed_column
                cf_list.append(f)
        col = crossed_column(cf_list, hash_bucket_size)
        wide_columns.append(col)
        wide_dim += hash_bucket_size
        if is_deep:
            deep_columns.append(embedding_column(col, dimension=embedding_dim(hash_bucket_size)))
            deep_dim += embedding_dim(hash_bucket_size)
    # add columns logging info
    tf.logging.info('Build total {} wide columns'.format(len(wide_columns)))
    for col in wide_columns:
        tf.logging.debug('Wide columns: {}'.format(col))
    tf.logging.debug('Build total {} deep columns'.format(len(deep_columns)))
    for col in deep_columns:
        tf.logging.info('Deep columns: {}'.format(col))
    tf.logging.info('Wide input dimension is : {}'.format(wide_dim))
    tf.logging.info('Deep input dimension is : {}'.format(deep_dim))
    return wide_columns, deep_columns


def _csv_column_defaults():
    """parse columns to record_defaults param in tf.decode_csv func
    Return: _CSV_COLUMN_DEFAULTS Ordereddict {'feature name': [''],...}
    """
    _CSV_COLUMN_DEFAULTS = OrderedDict()
    _CSV_COLUMN_DEFAULTS['label'] = [0]  # first label default, empty if the field is must
    feature_all = get_feature_name()
    feature_conf_dic = _read_feature_conf()
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
    feature_all = get_feature_name()
    feature_conf_dic = _read_feature_conf()
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


def _test():
    """test for above functions"""
    print('Input data schema:')
    data_schema = _read_data_schema()
    print(data_schema)

    feature_conf_dic = _read_feature_conf()
    print('\nFeature conf:')
    for f in feature_conf_dic.items():
        print(f)

    cross_feature_list = _read_cross_feature_conf()
    print('\nCross feature conf:')
    for f in cross_feature_list:
        print(f)

    category_feature = get_feature_name('category')
    print('\nCategory feature:'.format(category_feature))

    print(_csv_column_defaults())
    print(_column_to_dtype())


if __name__ == '__main__':
    _test()
    tf.logging.set_verbosity(tf.logging.INFO)
    build_model_columns()
