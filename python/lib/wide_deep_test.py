from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import sys
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)

from lib.read_conf import Config
from lib.dataset import input_fn
from lib.build_estimator import build_estimator


TEST_CSV = os.path.join(os.path.dirname(PACKAGE_DIR), 'data/test/test2')
USED_FEATURE_KEY = Config().get_feature_name('used')


def _read_test_input():
    for line in open(TEST_CSV):
        return line

TEST_INPUT_VALUES = _read_test_input()
TEST_INPUT_KEYS = Config().get_feature_name()
TEST_INPUT = zip(TEST_INPUT_KEYS, TEST_INPUT_VALUES)


class BaseTest(tf.test.TestCase):

    def setUp(self):
        # Create temporary CSV file
        self.temp_dir = self.get_temp_dir()
        self.input_csv = os.path.join(self.temp_dir, 'test.csv')
        with tf.gfile.Open(self.input_csv, 'w') as temp_csv:
            temp_csv.write(TEST_INPUT_VALUES)

    def test_input_fn(self):
        features, labels = input_fn(self.input_csv, 'eval', batch_size=1)
        with tf.Session() as sess:
            features, labels = sess.run((features, labels))
        # Compare the two features dictionaries.
        for key in USED_FEATURE_KEY:
            self.assertTrue(key in features)
            self.assertEqual(len(features[key]), 1)

            feature_value = features[key][0]
            # Convert from bytes to string for Python 3.
            if isinstance(feature_value, bytes):
                feature_value = feature_value.decode()
            self.assertEqual(TEST_INPUT[key], feature_value)
        self.assertFalse(labels)

    def build_and_test_estimator(self, model_type):
        """Ensure that model trains and minimizes loss."""
        model = build_estimator(self.temp_dir, model_type)

        # Train for 1 step to initialize model and evaluate initial loss
        model.train(
            input_fn=lambda: input_fn(
                TEST_CSV, None, 'eval', batch_size=1),
            steps=1)
        initial_results = model.evaluate(
            input_fn=lambda: input_fn(
                TEST_CSV, None, 'eval', batch_size=1))

        # Train for 100 epochs at batch size 3 and evaluate final loss
        model.train(
            input_fn=lambda: input_fn(
                TEST_CSV, None, 'eval', batch_size=8))
        final_results = model.evaluate(
            input_fn=lambda: input_fn(
                TEST_CSV, None, 'eval', batch_size=1))

        print('%s initial results:' % model_type, initial_results)
        print('%s final results:' % model_type, final_results)

        # Ensure loss has decreased, while accuracy and both AUCs have increased.
        self.assertLess(final_results['loss'], initial_results['loss'])
        self.assertGreater(final_results['auc'], initial_results['auc'])
        self.assertGreater(final_results['auc_precision_recall'],
                           initial_results['auc_precision_recall'])
        self.assertGreater(final_results['accuracy'], initial_results['accuracy'])

    def test_wide_deep_estimator_training(self):
        self.build_and_test_estimator('wide_deep')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.test.main()
