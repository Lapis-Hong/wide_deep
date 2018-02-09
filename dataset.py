#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/24
from collections import OrderedDict
import tensorflow as tf

from read_conf import Config


class Dataset(object):
    """A class to parse csv data and build input_fn for tf.estimators"""
    def __init__(self):
        self.config = Config()
        self.buffer_size = self.config.train["shuffle_buffer_size"]
        self.num_parallel_calls = self.config.train["num_parallel_calls"]
        self.multivalue = self.config.train["multivalue"]
        self.is_distribution = self.config.distribution["is_distribution"]
        cluster = self.config.distribution["cluster"]
        job_name = self.config.distribution["job_name"]
        task_index = self.config.distribution["task_index"]
        self.num_workers = 1 + len(cluster["worker"])  # must have 1 chief worker
        self.worker_index = task_index if job_name == "worker" else self.num_workers-1
        self.feature = self.config.get_feature_name()  # all features
        self.feature_used = self.config.get_feature_name('used')  # used features
        self.feature_unused = self.config.get_feature_name('unused')  # unused features
        self.feature_conf = self.config.read_feature_conf()  # feature conf dict
        self.csv_defaults = self._column_to_csv_defaults()
        self.csv_defaults_without_label = self.csv_defaults.copy()
        self.csv_defaults_without_label.pop('label')

    def _column_to_csv_defaults(self):
        """parse columns to record_defaults param in tf.decode_csv func
        Return: 
            _CSV_COLUMN_DEFAULTS OrderedDict {'feature name': [''],...}
        """
        _CSV_COLUMN_DEFAULTS = OrderedDict()
        _CSV_COLUMN_DEFAULTS['label'] = [0]  # first label default, empty if the field is must
        for f in self.feature:
            if f in self.feature_conf:  # used features
                conf = self.feature_conf[f]
                if conf['type'] == 'category':
                    if conf['transform'] == 'identity':  # identity category column need int type
                        _CSV_COLUMN_DEFAULTS[f] = [0]
                    else:
                        _CSV_COLUMN_DEFAULTS[f] = ['']
                else:
                    _CSV_COLUMN_DEFAULTS[f] = [0.0]  # 0.0 for float32
            else:  # unused features
                _CSV_COLUMN_DEFAULTS[f] = ['']
        return _CSV_COLUMN_DEFAULTS

    def _column_to_dtype(self):
        """Parse columns to tf.dtype
         Return: 
             similar to _csv_column_defaults()
         """
        _column_dtype_dic = OrderedDict()
        _column_dtype_dic['label'] = tf.int32
        for f in self.feature:
            if f in self.feature_conf:
                conf = self.feature_conf[f]
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

    def _parse_csv(self, value):  # value: Tensor("arg0:0", shape=(), dtype=string)
        columns = tf.decode_csv(value, record_defaults=self.csv_defaults.values(),
                                field_delim='\t', use_quote_delim=False, na_value='-')
        # na_value fill with record_defaults
        # `tf.decode_csv` return Tensor list:
        # <tf.Tensor 'DecodeCSV:60' shape=() dtype=string>  rank 0 Tensor
        # columns = (tf.expand_dims(col, 0) for col in columns)
        #  fix rank 0 error for dataset.padded_patch()
        features = dict(zip(self.csv_defaults.keys(), columns))
        for f in self.feature_unused:
            features.pop(f)  # remove unused features
        labels = features.pop('label')
        return features, tf.equal(labels, 1)

    def _parse_csv_pred(self, value):
        """parse prediction data without label"""
        columns = tf.decode_csv(value, record_defaults=self.csv_defaults_without_label.values(),
                                field_delim='\t', use_quote_delim=False, na_value='-')
        features = dict(zip(self.csv_defaults_without_label.keys(), columns))
        for f in self.feature_unused:
            features.pop(f)  # remove unused features
        return features

    def _parse_csv_multivalue(self, value):
        columns = tf.decode_csv(value, record_defaults=self.csv_defaults.values(),
                                field_delim='\t', use_quote_delim=False, na_value='-')
        features = {}
        for f, tensor in zip(self.csv_defaults.keys(), columns):
            if f in self.feature_unused:
                continue
            if isinstance(self.csv_defaults[f][0], str):
                # input must be rank 1, return SparseTensor
                # print(st.values)  # <tf.Tensor 'StringSplit_11:1' shape=(?,) dtype=string>
                features[f] = tf.string_split([tensor], ",").values  # tensor shape (?,)
            else:
                # features[f] = tensor  # error
                features[f] = tf.expand_dims(tensor, 0)  # change shape from () to (1,)
        labels = features.pop('label')
        return features, tf.equal(labels, 1)

    def _parse_csv_pred_multivalue(self, value):
        columns = tf.decode_csv(value, record_defaults=self.csv_defaults_without_label.values(),
                                field_delim='\t', na_value='-')
        features = {}
        for f, tensor in zip(self.csv_defaults.keys(), columns):
            if f in self.feature_unused:
                continue
            if isinstance(self.csv_defaults[f][0], str):
                features[f] = tf.string_split([tensor], ",").values
            else:
                features[f] = tf.expand_dims(tensor, 0)
        return features

    # TODO: read from hdfs
    def input_fn(self, data_file, num_epochs, batch_size, shuffle=True):
        """Input function for train or evaluation (with label).
        Args:
            data_file: can be both file or directory.
        Returns:
            (features, label) 
            `features` is a dictionary in which each value is a batch of values for
            that feature; `labels` is a batch of labels.
        """
        # check file exsits
        assert tf.gfile.Exists(data_file), (
            'train or test data file not found. Please make sure you have either '
            'default data_file or set both arguments --train_data and --test_data.')
        if tf.gfile.IsDirectory(data_file):
            data_file_list = [f for f in tf.gfile.ListDirectory(data_file) if not f.startswith('.')]
            data_file = [data_file+'/'+file_name for file_name in data_file_list]
        # Extract lines from input files using the Dataset API.
        dataset = tf.data.TextLineDataset(data_file)
        if self.is_distribution:  # allows each worker to read a unique subset.
            dataset = dataset.shard(self.num_workers, self.worker_index)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size, seed=123)  # set of Tensor object
        tf.logging.info('Parsing input files: {}'.format(data_file))
        # Use `Dataset.map()` to build a pair of a feature dictionary
        # and a label tensor for each example.
        if self.multivalue:  # Shuffle, repeat, and batch the examples.
            dataset = dataset.map(
                self._parse_csv_multivalue,
                num_parallel_calls=self.num_parallel_calls).repeat(num_epochs)
            padding_dic = {k: [None] for k in Config().get_feature_name('used')}
            dataset = dataset.padded_batch(batch_size, padded_shapes=(padding_dic, [None]))  # rank no change
        else:  # batch(): each element tensor must have exactly same shape, change rank 0 to rank 1
            dataset = dataset.map(
                self._parse_csv,
                num_parallel_calls=self.num_parallel_calls).repeat(num_epochs).batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()

    def pred_input_fn(self, data_file, batch_size, multivalue=False):
        """Input function for prediction.
        Args:
            data_file: can be both file or directory.
            multivalue: if csv has columns with multivalue, set True.
        Returns:
            (features, label) 
            `features` is a dictionary in which each value is a batch of values for
            that feature; `labels` is a batch of labels.
        """
        if tf.gfile.IsDirectory(data_file):
            data_file_list = [f for f in tf.gfile.ListDirectory(data_file) if not f.startswith('.')]
            data_file = [data_file + '/' + file_name for file_name in data_file_list]
        dataset = tf.data.TextLineDataset(data_file)
        tf.logging.info('Parsing input files: {}'.format(data_file))
        if multivalue:
            dataset = dataset.map(self._parse_csv_pred_multivalue)
            padding_dic = {k: [None] for k in self.feature_used}
            dataset.padded_batch(batch_size, padded_shapes=(padding_dic))
        else:
            dataset = dataset.map(self._parse_csv_pred).batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()

    def _parse_func_test(self, batch_size=5, multivalue=False):
        self.multivalue = multivalue
        print('batch size: {}; multivalue: {}'.format(batch_size, multivalue))
        sess = tf.InteractiveSession()
        features, labels = self.input_fn('./data/train/train2', 1,
                                         batch_size=batch_size, shuffle=False)
        print("features:\n {}".format(features))
        print("labels:\n {}".format(labels))
        for f, v in features.items()[:5]:
            print('{}: {}'.format(f, sess.run(v)))
        print('labels: {}'.format(sess.run(labels)))
        # graph = tf.get_default_graph()
        # operations = graph.get_operations()
        # sess = tf.InteractiveSession()
        # print(operations)
        # print(sess.run(operations))

    def _input_tensor_test(self, batch_size=5, multivalue=False):
        """test for categorical_column and cross_column input."""
        self.multivalue = multivalue
        sess = tf.InteractiveSession()
        features, labels = self.input_fn('./data/train/train1', 1, batch_size=batch_size)
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
    # print(Dataset().csv_defaults)
    # print(Dataset().csv_defaults_without_label)
    Dataset()._parse_func_test(multivalue=False)
    Dataset()._parse_func_test(multivalue=True)
    Dataset()._input_tensor_test(multivalue=False)
    Dataset()._input_tensor_test(multivalue=True)


