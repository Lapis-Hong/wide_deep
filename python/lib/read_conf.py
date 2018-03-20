#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/24
"""Read All Configuration from wide_deep/conf/*.yaml"""
import os
import yaml

from os.path import dirname, abspath

BASE_DIR = os.path.join(dirname(dirname(dirname(abspath(__file__)))), 'conf')
SCHEMA_CONF_FILE = 'schema.yaml'
DATA_PROCESS_CONF_FILE = 'data_process.yaml'
FEATURE_CONF_FILE = 'feature.yaml'
CROSS_FEATURE_CONF_FILE = 'cross_feature.yaml'
MODEL_CONF_FILE = 'model.yaml'
TRAIN_CONF_FILE = 'train.yaml'
SERVING_CONF_FILE = 'serving.yaml'


class Config(object):
    """Config class 
    Class attributes: config, train, distribution, model, runconfig
    Class methods: read_cross_feature_conf, read_feature_conf, get_feature_name
    """
    def __init__(self,
                 schema_conf_file=SCHEMA_CONF_FILE,
                 data_process_conf_file=DATA_PROCESS_CONF_FILE,
                 feature_conf_file=FEATURE_CONF_FILE,
                 cross_feature_conf_file=CROSS_FEATURE_CONF_FILE,
                 model_conf_file=MODEL_CONF_FILE,
                 train_conf_file=TRAIN_CONF_FILE,
                 serving_conf_file=SERVING_CONF_FILE):
        self._schema_conf_file = os.path.join(BASE_DIR, schema_conf_file)
        self._data_process_conf_file = os.path.join(BASE_DIR, data_process_conf_file)
        self._feature_conf_file = os.path.join(BASE_DIR, feature_conf_file)
        self._cross_feature_conf_file = os.path.join(BASE_DIR, cross_feature_conf_file)
        self._model_conf_file = os.path.join(BASE_DIR, model_conf_file)
        self._train_conf_file = os.path.join(BASE_DIR, train_conf_file)
        self._serving_conf_file = os.path.join(BASE_DIR, serving_conf_file)

    def read_schema(self):
        with open(self._schema_conf_file) as f:
            return {k: v.lower() for k, v in yaml.load(f).items()}

    def read_data_process_conf(self):
        with open(self._data_process_conf_file) as f:
            return yaml.load(f)

    @staticmethod
    def _check_feature_conf(feature, valid_feature_name, **kwargs):
        type_ = kwargs["type"]
        trans = kwargs["transform"]
        param = kwargs["parameter"]
        if type_ is None or trans is None or param is None:
            raise ValueError("All attributes are required in feature conf, "
                             "found empty value for feature `{}`".format(feature))
        if feature not in valid_feature_name:
            raise ValueError("Invalid feature name `{}` in feature conf, "
                             "must be consistent with schema conf".format(feature))
        assert type_ in {'category', 'continuous'}, (
            "Invalid type `{}` for feature `{}` in feature conf, "
            "must be 'category' or 'continuous'".format(type_, feature))
        # check transform and parameter
        if type_ == 'category':
            assert trans in {'hash_bucket', 'identity', 'vocab'}, (
                "Invalid transform `{}` for feature `{}` in feature conf, "
                "must be one of `hash_bucket`, `vocab`, `identity`.".format(trans, feature))
            if trans == 'hash_bucket' or trans == 'identity':
                if not isinstance(param, int):
                    raise TypeError('Invalid parameter `{}` for feature `{}` in feature conf, '
                                    '{} parameter must be an integer.'.format(param, feature, trans))
            elif trans == 'vocab':
                if not isinstance(param, (tuple, list)):
                    raise TypeError('Invalid parameter `{}` for feature `{}` in feature conf, '
                                    'vocab parameter must be a list.'.format(param, feature))

        else:
            mean, std, boundaries = param['mean'], param['std'], param['boundaries']
            assert trans in {'numeric', 'discretize'}, \
                "Invalid transform `{}` for feature `{}` in feature conf, " \
                "continuous feature transform must be `numeric` or `discretize`.".format(trans, feature)
            if mean and not isinstance(mean, (float, int)):
                raise TypeError('Invalid parameter `{}` for feature `{}` in feature conf, '
                                'parameter mean must be int or float.'.format(mean, feature))
            if std and (not isinstance(std, (float, int)) or std <= 0):
                    raise TypeError('Invalid parameter `{}` for feature `{}` in feature conf, '
                                    'parameter std must be a positive number.'.format(std, feature))
            if trans == 'discretize':
                if not isinstance(boundaries, (tuple, list)):
                    raise TypeError('Invalid parameter `{}` for feature `{}` in feature conf, '
                                    'discretize parameter must be a list.'.format(boundaries, feature))
                for v in boundaries:
                    assert isinstance(v, (int, float)), \
                        "Invalid parameter `{}` for feature `{}` in feature conf, " \
                        "discretize parameter element must be integer or float.".format(boundaries, feature)

    @staticmethod
    def _check_cross_feature_conf(features, valid_feature_name, **kwargs):
        features_list = [f.strip() for f in features.split('&')]
        hash_bucket_size = kwargs["hash_bucket_size"]
        is_deep = kwargs["is_deep"]
        assert len(features_list) > 1, (
            'Invalid cross feature name `{}` in cross feature conf,'
            'at least 2 features'.format(features))
        for f in features_list:
            if f not in valid_feature_name:
                raise ValueError("Invalid cross feature name `{}` in cross feature conf, "
                                 "must be consistent with feature conf".format(features))
        if hash_bucket_size:
            assert isinstance(hash_bucket_size, (int, float)), (
                'Invalid hash_bucket_size `{}` for features `{}` in cross feature conf, ' 
                'expected int or float'.format(hash_bucket_size, features))
        if is_deep:
            assert is_deep in {0, 1}, (
                'Invalid is_deep `{}` for features `{}`, ' 
                'expected 0 or 1.'.format(is_deep, features))

    def read_feature_conf(self):
        with open(self._feature_conf_file) as f:
            feature_conf = yaml.load(f)
            valid_feature_name = self.read_schema().values()
            for feature, conf in feature_conf.items():
                self._check_feature_conf(feature.lower(), valid_feature_name, **conf)
            return feature_conf

    def read_cross_feature_conf(self):
        with open(self._cross_feature_conf_file) as f:
            cross_feature_conf = yaml.load(f)
            conf_list = []
            valid_feature_name = self.read_feature_conf()  # used features
            for features, conf in cross_feature_conf.items():
                self._check_cross_feature_conf(features, valid_feature_name, **conf)
                features = [f.strip() for f in features.split('&')]
                hash_bucket_size = 1000*conf["hash_bucket_size"] or 10000  # defaults to 10k
                is_deep = conf["is_deep"] if conf["is_deep"] is not None else 1  # defaults to 10k
                conf_list.append((features, hash_bucket_size, is_deep))
            return conf_list

    def _read_model_conf(self):
        with open(self._model_conf_file) as f:
            return yaml.load(f)

    def _read_train_conf(self):
        with open(self._train_conf_file) as f:
            return yaml.load(f)

    def _read_serving_conf(self):
        with open(self._serving_conf_file) as f:
            return yaml.load(f)

    @property
    def config(self):
        return self._read_train_conf()

    @property
    def train(self):
        return self._read_train_conf()["train"]

    @property
    def distribution(self):
        return self._read_train_conf()["distribution"]

    @property
    def runconfig(self):
        return self._read_train_conf()["runconfig"]

    @property
    def model(self):
        return self._read_model_conf()

    @property
    def serving(self):
        return self._read_serving_conf()

    def get_feature_name(self, feature_type='all'):
        """
        Args:
         feature_type: one of {'all', 'used', 'category', 'continuous'}
        Return: feature name list
        """
        feature_conf_dic = self.read_feature_conf()
        feature_list = self.read_schema().values()
        feature_list.remove('clk')
        if feature_type == 'all':
            return feature_list
        elif feature_type == 'used':
            return feature_conf_dic.keys()
        elif feature_type == 'unused':
            return set(feature_list) - set(feature_conf_dic.keys())
        elif feature_type == 'category':
            return [feature for feature, conf in feature_conf_dic.items() if conf['type'] == 'category']
        elif feature_type == 'continuous':
            return [feature for feature, conf in feature_conf_dic.items() if conf['type'] == 'continuous']
        else:
            raise ValueError("Invalid parameter, must be one of 'all', 'used', 'category, 'continuous")


def _test():
    config = Config()
    """test for Config methods"""
    print('\nTrain config:')
    print(config.config)
    print(config.train)
    print(config.runconfig)
    print(config.train["model_dir"])

    print('\nModel conf:')
    for k, v in config.model.items():
        print(k, v)

    feature_conf_dic = config.read_feature_conf()
    print('\nFeature conf:')
    for k, v in feature_conf_dic.items():
        print(k, v)

    cross_feature_list = config.read_cross_feature_conf()
    print('\nCross feature conf:')
    for f in cross_feature_list:
        print(f)

    category_feature = config.get_feature_name('category')
    print('\nCategory feature:')
    print(category_feature)

    members = [m for m in Config.__dict__ if not m.startswith('_')]
    print('\nConfig class members:')
    print(members)

if __name__ == '__main__':
    _test()




