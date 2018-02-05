#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/2/2
"""Provide some utility function to the model"""
import time
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
