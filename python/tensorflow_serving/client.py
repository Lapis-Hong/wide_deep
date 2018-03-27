#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/3/19
"""A client that talks to tensorflow_model_server loaded with SavedModel.

Typical usage example:
    client.py --num_tests=100 --server=localhost:9000
"""

from __future__ import print_function

from os.path import dirname, abspath, join
import sys
import threading

import numpy as np
import tensorflow as tf

from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

PACKAGE_DIR = dirname(dirname(abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)
from lib.utils.util import column_to_dtype
from lib.read_conf import Config


tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('num_tests', 100, 'Number of test images')
tf.app.flags.DEFINE_string('server', 'localhost:9000', 'PredictionService host:port')
tf.app.flags.DEFINE_string('model', 'wide_deep',
                           'Model name.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS


class _ResultCounter(object):
    """Counter for the prediction results."""

    def __init__(self, num_tests, concurrency):
        self._num_tests = num_tests
        self._concurrency = concurrency
        self._error = 0
        self._done = 0
        self._active = 0
        self._condition = threading.Condition()

    def inc_error(self):
        with self._condition:
            self._error += 1

    def inc_done(self):
        with self._condition:
            self._done += 1
            self._condition.notify()

    def dec_active(self):
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def get_error_rate(self):
        with self._condition:
            while self._done != self._num_tests:
                self._condition.wait()
            return self._error / float(self._num_tests)

    def throttle(self):
        with self._condition:
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1


def _create_rpc_callback(label, result_counter):
    """Creates RPC callback function.
    Args:
        label: The correct label for the predicted example.
        result_counter: Counter for the prediction result.
    Returns:
        The callback function.
    """
    def _callback(result_future):
        """Callback function.
        Calculates the statistics for the prediction result.
        Args:
            result_future: Result future of the RPC.
        """
        exception = result_future.exception()
        if exception:
            result_counter.inc_error()
            print(exception)
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
            response = np.array(
                result_future.result().outputs['scores'].float_val)
            prediction = np.argmax(response)
            if label != prediction:
                result_counter.inc_error()
        result_counter.inc_done()
        result_counter.dec_active()
    return _callback


def do_inference(hostport, work_dir, concurrency, num_tests):
    """Tests PredictionService with concurrent requests.
    Args:
        hostport: Host:port address of the PredictionService.
        work_dir: The full path of working directory for test data set.
        concurrency: Maximum number of concurrent requests.
        num_tests: Number of test images to use.
    Returns:
        The classification error rate.
    Raises:
        IOError: An error occurred processing test data set.
    """
    test_data_set = mnist_input_data.read_data_sets(work_dir).test
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    result_counter = _ResultCounter(num_tests, concurrency)
    for _ in range(num_tests):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'mnist'
        request.model_spec.signature_name = 'predict_images'
        image, label = test_data_set.next_batch(1)
        request.inputs['images'].CopyFrom(
            tf.contrib.util.make_tensor_proto(image[0], shape=[1, image[0].size]))
        result_counter.throttle()
        result_future = stub.Predict.future(request, 5.0)  # 5 seconds
        result_future.add_done_callback(
            _create_rpc_callback(label[0], result_counter))
    return result_counter.get_error_rate()


PRED_FILE = join(dirname(dirname(dirname(abspath(__file__)))), 'data/pred/pred1')


def _read_test_input():
    for line in open(PRED_FILE):
        return line.strip('\n').split('\t')

# Example Features for a movie recommendation application:
#    feature {
#      key: "age"
#      value { float_list {
#        value: 29.0
#      }}
#    }
#    feature {
#      key: "movie"
#      value { bytes_list {
#        value: "The Shawshank Redemption"
#        value: "Fight Club"
#      }}
#    }


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def pred_input_fn(csv_data):
    """Prediction input fn for a single data, used for serving client"""
    conf = Config()
    feature = conf.get_feature_name()
    feature_unused = conf.get_feature_name('unused')
    feature_conf = conf.read_feature_conf()
    csv_default = column_to_dtype(feature, feature_conf)
    csv_default.pop('label')

    feature_dict = {}
    for idx, f in enumerate(csv_default.keys()):
        if f in feature_unused:
            continue
        else:
            if csv_default[f] == tf.string:
                feature_dict[f] = _bytes_feature(csv_data[idx])
            else:
                feature_dict[f] = _float_feature(float(csv_data[idx]))
    return feature_dict


def main(_):
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = FLAGS.model
    request.model_spec.signature_name = 'serving_default'
    # feature_dict = {'age': _float_feature(value=25),
    #               'capital_gain': _float_feature(value=0),
    #               'capital_loss': _float_feature(value=0),
    #               'education': _bytes_feature(value='11th'.encode()),
    #               'education_num': _float_feature(value=7),
    #               'gender': _bytes_feature(value='Male'.encode()),
    #               'hours_per_week': _float_feature(value=40),
    #               'native_country': _bytes_feature(value='United-States'.encode()),
    #               'occupation': _bytes_feature(value='Machine-op-inspct'.encode()),
    #               'relationship': _bytes_feature(value='Own-child'.encode()),
    #               'workclass': _bytes_feature(value='Private'.encode())}
    # label = 0
    data = _read_test_input()
    feature_dict = pred_input_fn(data)

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    serialized = example.SerializeToString()

    request.inputs['inputs'].CopyFrom(
        tf.contrib.util.make_tensor_proto(serialized, shape=[1]))

    result_future = stub.Predict.future(request, 5.0)
    prediction = result_future.result().outputs['scores']

    # print('True label: ' + str(label))
    print('Prediction: ' + str(np.argmax(prediction.float_val)))

if __name__ == '__main__':
    tf.app.run()
