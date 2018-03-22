#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/2/2
"""Provide some utility function."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
from collections import OrderedDict
from functools import wraps

import tensorflow as tf


def timer(info=''):
    """parameter decarotor"""
    def _timer(func):
        @wraps(func)
        def wrapper(*args, **kwargs):  # passing params to func
            s_time = time.time()
            func(*args, **kwargs)
            e_time = time.time()
            period = (e_time-s_time) / 60.0
            print(info + '-> elapsed time: %.3f minutes' % period)
        return wrapper
    return _timer


def elapse_time(start_time):
    return round((time.time()-start_time) / 60)


def list_files(input_data):
    """if input file is a dir, convert to a file path list
    Return:
         file path list
    """
    if tf.gfile.IsDirectory(input_data):
        file_name = [f for f in tf.gfile.ListDirectory(input_data) if not f.startswith('.')]
        return [input_data + '/' + f for f in file_name]
    else:
        return [input_data]


def record_dataset(filenames, height, width, depth):
    """Returns an input pipeline Dataset from `filenames`."""
    record_bytes = height * width * depth
    return tf.data.FixedLengthRecordDataset(filenames, record_bytes)


def get_filenames(data_dir):
    """Returns a list of filenames."""
    assert os.path.exists(data_dir), (
        'Image data dir {} not found.'.format(data_dir))
    return os.listdir(data_dir)


def column_to_dtype(feature, feature_conf):
    """Parse columns to tf.dtype
     Return: 
         similar to _csv_column_defaults()
     """
    _column_dtype_dic = OrderedDict()
    _column_dtype_dic['label'] = tf.int32
    for f in feature:
        if f in feature_conf:
            conf = feature_conf[f]
            if conf['type'] == 'category':
                if conf['transform'] == 'identity':  # identity category column need int type
                    _column_dtype_dic[f] = tf.int32
                else:
                    _column_dtype_dic[f] = tf.string
            else:
                _column_dtype_dic[f] = tf.float32  # 0.0 for float32
        else:
            _column_dtype_dic[f] = tf.string
    return _column_dtype_dic



