#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/15
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from read_conf import Config

CONFIG = Config()
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
    feature_conf_dic = Config.read_feature_conf()
    tf.logging.info('Total used feature class: {}'.format(len(feature_conf_dic)))
    cross_feature_list = Config.read_cross_feature_conf()
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
    tf.logging.info('Build total {} deep columns'.format(len(deep_columns)))
    for col in deep_columns:
        tf.logging.debug('Deep columns: {}'.format(col))
    tf.logging.info('Wide input dimension is: {}'.format(wide_dim))
    tf.logging.info('Deep input dimension is: {}'.format(deep_dim))
    return wide_columns, deep_columns


def build_estimator(model_dir, model_type):
    """Build an estimator appropriate for the given model type."""
    wide_columns, deep_columns = build_model_columns()
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
                learning_rate=CONFIG.wide_learning_rate,
                l1_regularization_strength=CONFIG.wide_l1,
                l2_regularization_strength=CONFIG.wide_l2),  # 'Ftrl',
            partitioner=None,
            config=run_config)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=CONFIG.hidden_units,
            optimizer=tf.train.ProximalAdagradOptimizer(
                learning_rate=CONFIG.deep_learning_rate,
                l1_regularization_strength=CONFIG.deep_l1,
                l2_regularization_strength=CONFIG.deep_l2),  # {'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'}
            activation_fn=tf.nn.relu,
            dropout=CONFIG.dropout,
            weight_column=None,
            input_layer_partitioner=None,
            config=run_config)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            linear_optimizer=tf.train.FtrlOptimizer(
                learning_rate=CONFIG.wide_learning_rate,
                l1_regularization_strength=CONFIG.wide_l1,
                l2_regularization_strength=CONFIG.wide_l2),
            dnn_feature_columns=deep_columns,
            dnn_optimizer=tf.train.ProximalAdagradOptimizer(
                learning_rate=CONFIG.deep_learning_rate,
                l1_regularization_strength=CONFIG.deep_l1,
                l2_regularization_strength=CONFIG.deep_l2),
            dnn_hidden_units=CONFIG.hidden_units,
            dnn_activation_fn=tf.nn.relu,
            dnn_dropout=CONFIG.dropout,
            n_classes=2,
            weight_column=None,
            label_vocabulary=None,
            input_layer_partitioner=None,
            config=run_config)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    build_model_columns()
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
