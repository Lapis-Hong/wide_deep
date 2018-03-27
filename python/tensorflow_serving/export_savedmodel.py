#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/3/19
"""Export TensorFlow SavedModel of tf.estimator.

SavedModel is a language-neutral, recoverable, hermetic serialization format. 
SavedModel enables higher-level systems and tools to produce, consume, and 
transform TensorFlow models. TensorFlow provides several mechanisms for interacting 
with SavedModel, including tf.saved_model APIs, Estimator APIs and a CLI.

To prepare a trained Estimator for serving, you must export it in the standard SavedModel format. 

https://www.tensorflow.org/programmers_guide/saved_model#using_savedmodel_with_estimators
"""
from __future__ import print_function

import os
import sys

import tensorflow as tf

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)

from lib.build_estimator import _build_model_columns, build_custom_estimator
from lib.read_conf import Config

model_base_dir = Config().train['model_dir']
CONF = Config().serving['SavedModel']

tf.app.flags.DEFINE_string('model_type', CONF['model_type'],
                           """Model type to export""")
tf.app.flags.DEFINE_string('checkpoint_path', CONF['checkpoint_path'],
                           """Directory to read training checkpoints. If None, use latest.""")
tf.app.flags.DEFINE_string('export_dir', CONF['model_dir'],
                           """Directory to export inference model.""")
tf.app.flags.DEFINE_integer('model_version', CONF['model_version'], 'version number of the model.')
FLAGS = tf.app.flags.FLAGS


def main(_):
    if FLAGS.model_version <= 0:
        print('Please specify a positive value for version number.')
        sys.exit(-1)
    # tf.estimator.export.build_parsing_serving_input_receiver_fn
    # tf.estimator.export.build_raw_serving_input_receiver_fn
    # If these utilities do not meet your needs, you are free to write your own serving_input_receiver_fn()

    # feature_spec = {
    #     "some_feature": tf.FixedLenFeature([], dtype=tf.string, default_value=""),
    #     "some_feature": tf.VarLenFeature(dtype=tf.string),
    # }
    #
    # def _serving_input_receiver_fn():
    #     serialized_tf_example = tf.placeholder(dtype=tf.string, shape=None,
    #                                            name='input_example_tensor')
    #     # key (e.g. 'examples') should be same with the inputKey when you
    #     # buid the request for prediction
    #     receiver_tensors = {'examples': serialized_tf_example}
    #     features = tf.parse_example(serialized_tf_example, your_feature_spec)
    #     return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
    # estimator.export_savedmodel(export_dir, _serving_input_receiver_fn)

    wide_columns, deep_columns = _build_model_columns()
    feature_columns = wide_columns + deep_columns
    # for f in feature_columns:
    #     print(f._parse_example_spec)
    # A dict mapping each feature key to a FixedLenFeature or VarLenFeature value.
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    serving_input_receiver_fn = \
        tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

    model_dir = os.path.join(model_base_dir, FLAGS.model_type)
    export_dir = os.path.join(FLAGS.export_dir, FLAGS.model_type)
    model = build_custom_estimator(model_dir, FLAGS.model_type)
    model.export_savedmodel(export_dir,
                            serving_input_receiver_fn,
                            as_text=CONF['as_text'],
                            checkpoint_path=FLAGS.checkpoint_path)

if __name__ == '__main__':
    tf.app.run()
