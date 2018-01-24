#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/24
from collections import OrderedDict


class Config(object):

    @staticmethod
    def _read_train_conf():
        """Read train configuration
        Return: Dict {parameter_name: parameter_value,...}
        """
        config = {}
        for line in open('./conf/train.conf'):
            line = line.strip().strip('\n')
            if line.startswith('#') or not line:
                continue
            key = line.split('=')[0].strip().lower()
            value = line.split('=')[1].strip().lower()
            config[key] = value or None
        return config

    @classmethod
    def __init__(cls):
        config = cls._read_train_conf()
        cls.config = config
        # train parameter
        cls.model_dir = config["model_dir"]
        cls.model_type = config["model_type"]
        cls.train_data = config["train_data"]
        cls.test_data = config["test_data"]
        cls.train_epochs = config["train_epochs"]
        cls.batch_size = int(config["batch_size"])
        cls.keep_train = bool(config["keep_train"])
        cls.is_distribution = bool(config["is_distribution"])
        cls.shuffle_buffer_size = int(config["shuffle_buffer_size"])
        # model hyperparameter
        cls.hidden_units = map(int, config["hidden_units"].split(','))
        cls.wide_learning_rate = float(config["wide_learning_rate"])
        cls.deep_learning_rate = float(config["deep_learning_rate"])
        cls.wide_l1 = float(config["wide_l1"]) if config["wide_l1"] is not None else None
        cls.wide_l2 = float(config["wide_l2"]) if config["wide_l2"] is not None else None
        cls.deep_l1 = float(config["deep_l1"]) if config["deep_l1"] is not None else None
        cls.deep_l2 = float(config["deep_l2"]) if config["deep_l1"] is not None else None
        cls.dropout = float(config["dropout"]) if config["dropout"] is not None else None

    @staticmethod
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

    @classmethod
    def read_feature_conf(cls):
        """Read feature configuration, parse all used features conf
        Usage: 
        feature_conf_dic = read_feature_conf()
        f_param = feature_conf_dic['request_id'].get('feature_parammeter'))
        Return: 
            {'request_id', {'feature_type': 'category', 'feature_transform': 'hash_bucket', 'feature_parameter': [500000]}, ...}"""
        feature_schema = cls._read_data_schema()[1:]
        feature_conf_dic = OrderedDict()
        for nu, line in enumerate(open('./conf/feature_default.conf')):
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

    @classmethod
    def read_cross_feature_conf(cls):
        """Read cross feature configuration
        Return: list with tuple element, [(['adplan_id', 'category'], 10000, 1), (),...]
        """
        feature_used = cls.read_feature_conf().keys()
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
            # hash_bucket_size = 1000*int(hash_bucket_size.replace('k', '')) if hash_bucket_size else 10000
            hash_bucket_size = int(1000 * float(hash_bucket_size.replace('k', '') or 10))  # default 10k
            # is_deep = int(is_deep) if is_deep else 1  # default 1
            is_deep = int(is_deep or 1)  # default 1
            cross_feature_list.append((cross_feature, hash_bucket_size, is_deep))
        return cross_feature_list

    @classmethod
    def get_feature_name(cls, feature_type='all'):
        """
        Args:
         feature_type: one of {'all', 'used', 'category', 'continuous'}
        Return: feature name list
        """
        feature_conf_dic = cls.read_feature_conf()
        if feature_type == 'all':
            return cls._read_data_schema()[1:]
        elif feature_type == 'used':
            print(feature_conf_dic.keys())
            return feature_conf_dic.keys()
        elif feature_type == 'category':
            return [feature for feature, conf in feature_conf_dic.items() if conf['feature_type'] == 'category']
        elif feature_type == 'continuous':
            return [feature for feature, conf in feature_conf_dic.items() if conf['feature_type'] == 'continuous']
        else:
            raise TypeError("Invalid parameter, must be one of 'all', 'used', 'category, 'continuous")


def _test():
    config = Config()
    print(config.hidden_units)
    print(config.dropout)
    """test for Config methods"""
    print('Input data schema:')
    data_schema = Config._read_data_schema()
    print(data_schema)

    feature_conf_dic = Config.read_feature_conf()
    print('\nFeature conf:')
    for k, v in feature_conf_dic.items():
        print(k, v)

    cross_feature_list = Config.read_cross_feature_conf()
    print('\nCross feature conf:')
    for f in cross_feature_list:
        print(f)

    train_conf_dic = Config._read_train_conf()
    print('\nTrain conf:')
    for k, v in train_conf_dic.items():
        print(k, v)

    category_feature = Config.get_feature_name('category')
    print('\nCategory feature:')
    print(category_feature)


if __name__ == '__main__':
    _test()
