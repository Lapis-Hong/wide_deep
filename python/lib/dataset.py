#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/24
"""Parse data and generate input_fn for tf.estimators"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import abc
import tensorflow as tf

import os
import sys
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)

from lib.read_conf import Config
from lib.utils import image_preprocessing, vgg_preprocessing


class _CTRDataset(object):
    """Interface for dataset using abstract class"""
    __metaclass__ = abc.ABCMeta

    def __init__(self, data_file):
        # check file exsits, turn to list so that data_file can be both file or directory.
        assert tf.gfile.Exists(data_file), (
            'data file: {} not found. Please check input data path'.format(data_file))
        if tf.gfile.IsDirectory(data_file):
            data_file_list = [f for f in tf.gfile.ListDirectory(data_file) if not f.startswith('.')]
            data_file = [data_file + '/' + file_name for file_name in data_file_list]
        self._data_file = data_file
        self._conf = Config()
        self._train_conf = self._conf.train
        self._dist_conf = self._conf.distribution
        self._cnn_conf = self._conf.model
        self._shuffle_buffer_size = self._train_conf["num_examples"]
        self._num_parallel_calls = self._train_conf["num_parallel_calls"]
        self._train_epochs = self._train_conf["train_epochs"]

    @abc.abstractmethod
    def input_fn(self, mode, batch_size):
        """
        Abstract input function for train or evaluation (with label),
        abstract method must be implemented in subclasses when instantiate.
        Args:
            mode: `train`, `eval` or `pred`
                train for train mode, do shuffle, repeat num_epochs
                eval for eval mode, no shuffle, no repeat
                pred for pred input_fn, no shuffle, no repeat and no label 
            batch_size: Int
        Returns:
            (features, label) 
            `features` is a dictionary in which each value is a batch of values for
            that feature; `labels` is a batch of labels.
        """
        raise NotImplementedError('Calling an abstract method.')


class _CsvDataset(_CTRDataset):
    """A class to parse csv data and build input_fn for tf.estimators"""

    def __init__(self, data_file):
        super(_CsvDataset, self).__init__(data_file)
        self._pos_sample_loss_weight = self._train_conf["pos_sample_loss_weight"]
        self._neg_sample_loss_weight = self._train_conf["neg_sample_loss_weight"]
        self._use_weight = False
        if self._pos_sample_loss_weight is not None and self._neg_sample_loss_weight is not None:
            self._use_weight = True
            print("Using weight column to use weighted loss.")
        self._multivalue = self._train_conf["multivalue"]
        self._is_distribution = self._dist_conf["is_distribution"]
        cluster = self._dist_conf["cluster"]
        job_name = self._dist_conf["job_name"]
        task_index = self._dist_conf["task_index"]
        self._num_workers = 1 + len(cluster["worker"])  # must have 1 chief worker
        self._worker_index = task_index if job_name == "worker" else self._num_workers-1
        self._feature = self._conf.get_feature_name()  # all features
        self._feature_used = self._conf.get_feature_name('used')  # used features
        self._feature_unused = self._conf.get_feature_name('unused')  # unused features
        self._feature_conf = self._conf.read_feature_conf()  # feature conf dict
        self._csv_defaults = self._column_to_csv_defaults()

    def _column_to_csv_defaults(self):
        """parse columns to record_defaults param in tf.decode_csv func
        Return: 
            OrderedDict {'feature name': [''],...}
        """
        csv_defaults = OrderedDict()
        csv_defaults['label'] = [0]  # first label default, empty if the field is must
        for f in self._feature:
            if f in self._feature_conf:  # used features
                conf = self._feature_conf[f]
                if conf['type'] == 'category':
                    if conf['transform'] == 'identity':  # identity category column need int type
                        csv_defaults[f] = [0]
                    else:
                        csv_defaults[f] = ['']
                else:
                    csv_defaults[f] = [0.0]  # 0.0 for float32
            else:  # unused features
                csv_defaults[f] = ['']
        return csv_defaults

    def _parse_csv(self, is_pred=False, field_delim='\t', na_value='-', multivalue_delim=','):
        """Parse function for csv data
        Args:
            is_pred: bool, defaults to False
                True for pred mode, parse input data with label
                False for train or eval mode, parse input data without label
            field_delim: csv fields delimiter, defaults to `\t`
            na_value: use csv defaults to fill na_value
            multivalue: bool, defaults to False
                True for csv data with multivalue features.
                eg:   f1       f2   ...
                    a, b, c    1    ...
                     a, c      2    ...
                     b, c      0    ...
            multivalue_delim: multivalue feature delimiter, defaults to `,`
        Returns:
            feature dict: {feature: Tensor ... }
        """
        if is_pred:
            self._csv_defaults.pop('label')
        csv_defaults = self._csv_defaults
        multivalue = self._multivalue
        pos_w = self._pos_sample_loss_weight
        neg_w = self._neg_sample_loss_weight
        use_weight = self._use_weight

        def parser(value):
            """Parse train and eval data with label
            Args:
                value: Tensor("arg0:0", shape=(), dtype=string)
            """
            # `tf.decode_csv` return rank 0 Tensor list: <tf.Tensor 'DecodeCSV:60' shape=() dtype=string>
            # na_value fill with record_defaults
            columns = tf.decode_csv(
                value, record_defaults=csv_defaults.values(),
                field_delim=field_delim, use_quote_delim=False, na_value=na_value)
            features = dict(zip(csv_defaults.keys(), columns))
            for f, tensor in features.items():
                if f in self._feature_unused:
                    features.pop(f)  # remove unused features
                    continue
                if multivalue:  # split tensor
                    if isinstance(csv_defaults[f][0], str):
                        # input must be rank 1, return SparseTensor
                        # print(st.values)  # <tf.Tensor 'StringSplit_11:1' shape=(?,) dtype=string>
                        features[f] = tf.string_split([tensor], multivalue_delim).values  # tensor shape (?,)
                    else:
                        features[f] = tf.expand_dims(tensor, 0)  # change shape from () to (1,)
            if is_pred:
                return features
            else:
                labels = tf.equal(features.pop('label'), 1)
                if use_weight:
                    pred = labels[0] if multivalue else labels  # pred must be rank 0 scalar
                    pos_weight, neg_weight = pos_w or 1, neg_w or 1
                    weight = tf.cond(pred, lambda: pos_weight, lambda: neg_weight)
                    features["weight_column"] = [weight]  # padded_batch need rank 1
                return features, labels
        return parser

    def input_fn(self, mode, batch_size):
        assert mode in {'train', 'eval', 'pred'}, (
            'mode must in `train`, `eval`, or `pred`, found {}'.format(mode))
        tf.logging.info('Parsing input csv files: {}'.format(self._data_file))
        # Extract lines from input files using the Dataset API.
        dataset = tf.data.TextLineDataset(self._data_file)
        if self._is_distribution:  # allows each worker to read a unique subset.
            dataset = dataset.shard(self._num_workers, self._worker_index)
        # Use `Dataset.map()` to build a pair of a feature dictionary
        # and a label tensor for each example.
        # Shuffle, repeat, and batch the examples.
        dataset = dataset.map(
            self._parse_csv(is_pred=(mode == 'pred')),
            num_parallel_calls=self._num_parallel_calls)
        if mode == 'train':
            dataset = dataset.shuffle(buffer_size=self._shuffle_buffer_size, seed=123)
            # dataset = dataset.repeat(self._train_epochs)  # define outside loop

        dataset = dataset.prefetch(2 * batch_size)
        if self._multivalue:
            padding_dic = {k: [None] for k in self._feature_used}
            if self._use_weight:
                padding_dic['weight_column'] = [None]
            padded_shapes = padding_dic if mode == 'pred' else (padding_dic, [None])
            dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
        else:
            # batch(): each element tensor must have exactly same shape, change rank 0 to rank 1
            dataset = dataset.batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()


class _ImageDataSet(_CTRDataset):
    """A class to parse image data and build input_fn for tf.estimators
    data only contains image (no label), so there is no need of pred version.
    TODO: debug and improve.
    """
    def __init__(self, data_file):
        super(_ImageDataSet, self).__init__(data_file)
        print(self._cnn_conf['cnn_height'])
        self._height = self._cnn_conf['cnn_height']
        self._width = self._cnn_conf['cnn_width']
        self._num_channels = self._cnn_conf['cnn_num_channels']
        self._weight_decay = self._cnn_conf['cnn_weight_decay']
        self._momentum = self._cnn_conf['cnn_momentum']
        self._use_distortion = self._cnn_conf['cnn_use_distortion']

    def parse_example(self, serialized_example, is_training, preprocess='custom'):
        """Parses a single tf.Example into image tensors.
        Args:
            preprocess: 'custom' or 'vgg'
                custom: custom image preprocessing, see utils.image_preprocessing
                vgg: standard vgg preprocessing, see utils.vgg_preprocessing
        Returns:
            feature dict {'image': Tensor}
        """
        assert preprocess in {'custom', 'vgg'}, 'Invalid preprocess parameters {}, must be `custom` or `vgg`'.format(preprocess)
        features = tf.parse_single_example(
            serialized_example,
                features={
                    'image': tf.FixedLenFeature([], tf.string),
                    # 'label': tf.FixedLenFeature([], tf.int64),
                    })
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([self._num_channels * self._height * self._width])
        # Reshape from [depth * height * width] to [depth, height, width].
        image = tf.cast(
            tf.transpose(tf.reshape(image, [self._num_channels, self._height, self._width]), [1, 2, 0]),
            tf.float32)
        if self._use_distortion:
            if preprocess == 'custom':
                image = image_preprocessing.preprocess_image(
                    image,
                    height=self._height,
                    width=self._width,
                    depth=self._num_channels,
                    is_training=is_training)
            else:
                image = vgg_preprocessing.preprocess_image(
                    image=image,
                    output_height=self._height,
                    output_width=self._width,
                    is_training=is_training)
        return {'image': image}

    # def parse_value(self, value, is_training):
    #     """Parse an Image record from `value`."""
    #     keys_to_features = {
    #         'image': tf.FixedLenFeature([], tf.string, default_value='')}
    #     parsed = tf.parse_single_example(value, keys_to_features)
    #     image = tf.image.decode_image(
    #         tf.reshape(parsed['image'], shape=[]),
    #         self._num_channels)
    #     image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    #     return image

    # def parse_raw(self, raw_record):
    #     """Parse image from a raw record."""
    #     # Every record consists of a image, with a fixed number of bytes for each.
    #     record_vector = tf.decode_raw(raw_record, tf.uint8)
    #     # reshape image from [depth * height * width] to [depth, height, width].
    #     depth_major = tf.reshape(record_vector, [self._num_channels, self._height, self._width])
    #     # Convert from [depth, height, width] to [height, width, depth]
    #     image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
    #     return image
    def input_fn(self, mode, batch_size):
        assert mode in {'train', 'eval', 'pred'}, (
            'mode must in `train`, `eval`, or `pred`, found {}'.format(mode))
        # dataset = tf.data.Dataset.from_tensor_slices([data_dir])  # multiple input data file
        # if is_training:
        #     dataset = dataset.shuffle(buffer_size=100)
        # dataset = dataset.flat_map(tf.data.TFRecordDataset)
        tf.logging.info('Parsing input image data files: {}'.format(self._data_file))
        dataset = tf.data.TFRecordDataset(self._data_file)
        dataset = dataset.map(lambda value: self.parse_example(value, mode == 'train'))
        dataset = dataset.prefetch(2*batch_size)
        if mode == 'train':
            # When choosing shuffle buffer sizes, larger sizes result in better
            # randomness, while smaller sizes have better performance.
            # seed must be same with above CsvDataset
            dataset = dataset.shuffle(buffer_size=self._shuffle_buffer_size, seed=123)
            # dataset = dataset.repeat(self._train_epochs)
        dataset = dataset.batch(batch_size)
        images = dataset.make_one_shot_iterator().get_next()
        return images


def input_fn(csv_data_file, img_data_file, mode, batch_size):
    """Combine input_fn for tf.estimators
    Combine both csv and image data; combine both train and pred mode.
    set img_data_file None to use only csv data
    """
    if mode == 'pred':
        features = _CsvDataset(csv_data_file).input_fn(mode, batch_size)
        if img_data_file is not None:
            img_data = _ImageDataSet(img_data_file).input_fn(mode, batch_size)
            features.update(img_data)  # add image Tensor to feature dict.
        return features

    else:
        features, label = _CsvDataset(csv_data_file).input_fn(mode, batch_size)
        if img_data_file is not None:
            img_data = _ImageDataSet(img_data_file).input_fn(mode, batch_size)
            features.update(img_data)  # add image Tensor to feature dict.
        return features, label


def _input_tensor_test(data_file, batch_size=5):
    """test for categorical_column and cross_column input."""
    sess = tf.InteractiveSession()
    features, labels = _CsvDataset(data_file).input_fn('train', batch_size=batch_size)
    print(features['ucomp'].eval())
    print(features['city_id'].eval())
    # categorical_column* can handle multivalue feature as a multihot
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
    csv_path = '../../data/train/train1'
    img_path = '../../data/image/train.tfrecords'
    _input_tensor_test(csv_path)
    sess = tf.InteractiveSession()
    data = input_fn(csv_path, img_path, 'train', 5)
    print(sess.run(data))






