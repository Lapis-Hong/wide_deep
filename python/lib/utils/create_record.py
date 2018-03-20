#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/3/6
"""This script convert the raw image to tfrecords format."""
import os
import tensorflow as tf
from PIL import Image

inpath = '../../data/image/train'
outpath = '../../data/image/train.tfrecords'


def create_record():
    writer = tf.python_io.TFRecordWriter(outpath)
    for img_name in os.listdir(inpath):
        img_path = os.path.join(inpath, img_name)
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img_raw = img.tobytes()
        example = tf.train.Example(
           features=tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
           }))
        writer.write(example.SerializeToString())
    writer.close()


def input_fn():
    # Transforms a scalar string `example_proto` into a pair of a scalar string and
    # a scalar integer, representing an image and its label, respectively.
    def _parse_function(example_proto):
        features = {"image": tf.FixedLenFeature((), tf.string, default_value="")}
        parsed = tf.parse_single_example(example_proto, features)
        # Perform additional preprocessing on the parsed data.
        image = tf.decode_raw(parsed["image"], tf.uint8)
        image = tf.reshape(image, [224, 224, 3])
        return image
    # Creates a dataset that reads all of the examples from two files, and extracts
    # the image and label features.
    dataset = tf.data.TFRecordDataset(outpath)
    dataset = dataset.map(_parse_function)
    img = dataset.make_one_shot_iterator().get_next()
    return img


if __name__ == '__main__':
    create_record()
    image = input_fn()
    sess = tf.InteractiveSession()
    print(sess.run(image))

