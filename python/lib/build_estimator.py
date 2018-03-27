#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/15
"""
Build feature columns using tf.feature_column API.
Build estimator using tf.estimator API and custom API (defined in lib module)
Use function `build_estimator` to use official classifier
Use function `build_costum_estimator` to use custom classifier.
"""
# # TODO: optimizer config, tf.train.optimizer choose which one ?
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import numpy as np
import tensorflow as tf

# fix ImportError: No mudule named lib.*
import sys
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)

from lib.dnn import DNN, MultiDNNClassifier
from lib.joint import WideAndDeepClassifier
from lib.read_conf import Config

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

config = Config()
CONF_WIDE = config.model['linear']
CONF_DNN = config.model['dnn']
CONF_CNN = config.model['cnn']


def _build_model_columns():
    """
    Build wide and deep feature columns from custom feature conf using tf.feature_column API
    wide_columns: category features + cross_features + [discretized continuous features]
    deep_columns: continuous features + category features(onehot or embedding for sparse features) + [cross_features(embedding)]
    Return: 
        _CategoricalColumn and __DenseColumn instance in tf.feature_column API
    """
    def embedding_dim(dim):
        """empirical embedding dim"""
        return int(np.power(2, np.ceil(np.log(dim**0.25))))

    def normalizer_fn_builder(scaler, *args):
        """normalizer_fn builder"""
        if scaler == 'min_max':
            return lambda x: (x-args[0]) / (x-args[1])
        elif scaler == 'standard':
            return lambda x: (x-args[0]) / args[1]
        else:
            return lambda x: tf.log(x)

    feature_conf_dic = config.read_feature_conf()
    cross_feature_list = config.read_cross_feature_conf()
    tf.logging.info('Total used feature class: {}'.format(len(feature_conf_dic)))
    tf.logging.info('Total used cross feature class: {}'.format(len(cross_feature_list)))

    wide_columns = []
    deep_columns = []
    wide_dim = 0
    deep_dim = 0
    for feature, conf in feature_conf_dic.items():
        f_type, f_tran, f_param = conf["type"], conf["transform"], conf["parameter"]
        if f_type == 'category':

            if f_tran == 'hash_bucket':
                hash_bucket_size = f_param
                embed_dim = embedding_dim(hash_bucket_size)
                col = categorical_column_with_hash_bucket(feature,
                    hash_bucket_size=hash_bucket_size,
                    dtype=tf.string)
                wide_columns.append(col)
                deep_columns.append(embedding_column(col,
                    dimension=embed_dim,
                    combiner='mean',
                    initializer=None,
                    ckpt_to_load_from=None,
                    tensor_name_in_ckpt=None,
                    max_norm=None,
                    trainable=True))
                wide_dim += hash_bucket_size
                deep_dim += embed_dim

            elif f_tran == 'vocab':
                col = categorical_column_with_vocabulary_list(feature,
                    vocabulary_list=map(str, f_param),
                    dtype=None,
                    default_value=-1,
                    num_oov_buckets=0)  # len(vocab)+num_oov_buckets
                wide_columns.append(col)
                deep_columns.append(indicator_column(col))
                wide_dim += len(f_param)
                deep_dim += len(f_param)

            elif f_tran == 'identity':
                num_buckets = f_param
                col = categorical_column_with_identity(feature,
                    num_buckets=num_buckets,
                    default_value=0)  # Values outside range will result in default_value if specified, otherwise it will fail.
                wide_columns.append(col)
                deep_columns.append(indicator_column(col))
                wide_dim += num_buckets
                deep_dim += num_buckets
        else:
            normalizaton, boundaries = f_param["normalization"], f_param["boundaries"]
            if f_tran is None:
                normalizer_fn = None
            else:
                normalizer_fn = normalizer_fn_builder(f_tran, normalizaton)
            col = numeric_column(feature,
                 shape=(1,),
                 default_value=0,  # default None will fail if an example does not contain this column.
                 dtype=tf.float32,
                 normalizer_fn=normalizer_fn)
            if boundaries:  # whether include continuous features in wide part
                wide_columns.append(bucketized_column(col, boundaries=boundaries))
                wide_dim += (len(boundaries)+1)
            deep_columns.append(col)
            deep_dim += 1

    for cross_features, hash_bucket_size, is_deep in cross_feature_list:
        cf_list = []
        for f in cross_features:
            f_type = feature_conf_dic[f]["type"]
            f_tran = feature_conf_dic[f]["transform"]
            f_param = feature_conf_dic[f]["parameter"]
            if f_type == 'continuous':
                cf_list.append(bucketized_column(numeric_column(f, default_value=0), boundaries=f_param['boundaries']))
            else:
                if f_tran == 'identity':
                    # If an input feature is of numeric type, you can use categorical_column_with_identity
                    cf_list.append(categorical_column_with_identity(f, num_buckets=f_param,
                    default_value=0))
                else:
                    cf_list.append(f)  # category col put the name in crossed_column
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


def _build_distribution():
    """Build distribution configuration variable TF_CONFIG in tf.estimator API"""
    TF_CONFIG = config.distribution
    if TF_CONFIG["is_distribution"]:
        cluster_spec = TF_CONFIG["cluster"]
        job_name = TF_CONFIG["job_name"]
        task_index = TF_CONFIG["task_index"]
        os.environ['TF_CONFIG'] = json.dumps(
            {'cluster': cluster_spec,
             'task': {'type': job_name, 'index': task_index}})
        run_config = tf.estimator.RunConfig()
        if job_name in ["ps", "chief", "worker"]:
            assert run_config.master == 'grpc://' + cluster_spec[job_name][task_index]  # grpc://10.120.180.212
            assert run_config.task_type == job_name
            assert run_config.task_id == task_index
            assert run_config.num_ps_replicas == len(cluster_spec["ps"])
            assert run_config.num_worker_replicas == len(cluster_spec["worker"]) + len(cluster_spec["chief"])
            assert run_config.is_chief == (job_name == "chief")
        elif job_name == "evaluator":
            assert run_config.master == ''
            assert run_config.evaluator_master == ''
            assert run_config.task_id == 0
            assert run_config.num_ps_replicas == 0
            assert run_config.num_worker_replicas == 0
            assert run_config.cluster_spec == {}
            assert run_config.task_type == 'evaluator'
            assert not run_config.is_chief


def build_estimator(model_dir, model_type):
    """Build an estimator using official tf.estimator API.
    Args:
        model_dir: model save directory
        model_type: one of {`wide`, `deep`, `wide_deep`}
    Returns:
        model instance of tf.estimator.Estimator class
    """
    wide_columns, deep_columns = _build_model_columns()
    _build_distribution()
    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig(**config.runconfig).replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))

    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns,
            weight_column=None,
            optimizer=tf.train.FtrlOptimizer(
                learning_rate=CONF_WIDE["wide_learning_rate"],
                l1_regularization_strength=CONF_WIDE["wide_l1"],
                l2_regularization_strength=CONF_WIDE["wide_l2"]),  # 'Ftrl',
            partitioner=None,
            config=run_config)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=CONF_DNN["hidden_units"],
            optimizer=tf.train.ProximalAdagradOptimizer(
                learning_rate=CONF_DNN["deep_learning_rate"],
                l1_regularization_strength=CONF_DNN["deep_l1"],
                l2_regularization_strength=CONF_DNN["deep_l2"]),  # {'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'}
            activation_fn=eval(CONF_DNN["activation_function"]),  # tf.nn.relu vs 'tf.nn.relu'
            dropout=CONF_DNN["dropout"],
            weight_column=None,
            input_layer_partitioner=None,
            config=run_config)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,  # self._model_dir = model_dir or self._config.model_dir
            linear_feature_columns=wide_columns,
            linear_optimizer=tf.train.FtrlOptimizer(
                learning_rate=CONF_WIDE["wide_learning_rate"],
                l1_regularization_strength=CONF_WIDE["wide_l1"],
                l2_regularization_strength=CONF_WIDE["wide_l2"]),
            dnn_feature_columns=deep_columns,
            dnn_optimizer=tf.train.ProximalAdagradOptimizer(
                learning_rate=CONF_DNN["deep_learning_rate"],
                l1_regularization_strength=CONF_DNN["deep_l1"],
                l2_regularization_strength=CONF_DNN["deep_l2"]),
            dnn_hidden_units=CONF_DNN["hidden_units"],
            dnn_activation_fn=eval(CONF_DNN["activation_function"]),
            dnn_dropout=CONF_DNN["dropout"],
            n_classes=2,
            weight_column=None,
            label_vocabulary=None,
            input_layer_partitioner=None,
            config=run_config)


def build_custom_estimator(model_dir, model_type):
    """Build an estimator using custom WideAndDeepClassifier API.
    Args:
        model_dir: model save directory
        model_type: one of {`wide`, `deep`, `wide_deep`}
    Returns:
        model instance of tf.estimator.Estimator class
    """
    wide_columns, deep_columns = _build_model_columns()
    _build_distribution()
    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig(**config.runconfig).replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))

    return WideAndDeepClassifier(
        model_type=model_type,
        model_dir=model_dir,
        with_cnn=CONF_CNN["use_flag"],
        cnn_optimizer=CONF_CNN["cnn_optimizer"],
        linear_feature_columns=wide_columns,
        linear_optimizer=CONF_WIDE["linear_optimizer"],
        dnn_feature_columns=deep_columns,
        dnn_optimizer=CONF_DNN["dnn_optimizer"],
        dnn_hidden_units=CONF_DNN["hidden_units"],
        dnn_connected_mode=CONF_DNN["connected_mode"],
        dnn_activation_fn=eval(CONF_DNN["activation_function"]),
        dnn_dropout=CONF_DNN["dropout"],
        dnn_batch_norm=CONF_DNN["batch_normalization"],
        n_classes=2,
        weight_column=None,
        label_vocabulary=None,
        input_layer_partitioner=None,
        config=run_config)


def _build_custom_estimator_test(model_dir, model_type):
    """Custom Classifier API test"""
    wide_columns, deep_columns = _build_model_columns()
    _build_distribution()
    run_config = tf.estimator.RunConfig(**config.runconfig).replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))
    dnn_1 = DNN(hidden_units=[16])
    dnn_2 = DNN(hidden_units=[32, 16])
    dnn_3 = DNN(hidden_units=[64, 32, 16])

    return MultiDNNClassifier(
        model_dir=model_dir,
        model_collections=(dnn_1, dnn_2, dnn_3),
        feature_columns=deep_columns,
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=CONF_DNN["deep_learning_rate"],
            l1_regularization_strength=CONF_DNN["deep_l1"],
            l2_regularization_strength=CONF_DNN["deep_l2"]),  # {'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'}
        weight_column=None,
        input_layer_partitioner=None,
        config=run_config)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    # _build_model_columns()
    # _build_distribution()
    model = build_estimator('../model', 'wide')
    # print(model.config)  # <tensorflow.python.estimator.run_config.RunConfig object at 0x118de4e10>
    # print(model.model_dir)  # ../model
    # print(model.model_fn)  # <function public_model_fn at 0x118de7b18>
    # print(model.params)  # {}
    # print(model.get_variable_names())
    # print(model.get_variable_value('dnn/hiddenlayer_0/bias'))
    # print(model.get_variable_value('dnn/hiddenlayer_0/bias/Adagrad'))
    # print(model.get_variable_value('dnn/hiddenlayer_0/kernel'))
    # print(model.latest_checkpoint())  # another 4 method is export_savedmodel,train evaluate predict
    # _build_custom_estimator_test('../model', 'wide')
    model = build_custom_estimator('../model', 'wide')