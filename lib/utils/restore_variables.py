#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/3/7
"""A simple script for inspect checkpoint files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("file_name", "", "Checkpoint filename")
tf.app.flags.DEFINE_string("tensor_name", "", "Name of the tensor to inspect")


checkpoint_path = os.path.join(model_dir, "model.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key)) # Remove this is you want to print only variable names


def print_tensors_in_checkpoint_file(file_name, tensor_name):
    """Prints tensors in a checkpoint file. 
    If no `tensor_name` is provided, prints the tensor names and shapes 
    in the checkpoint file. 
    If `tensor_name` is provided, prints the content of the tensor. 
    Args: 
      file_name: Name of the checkpoint file. 
      tensor_name: Name of the tensor in the checkpoint file to print. 
    """
    try:
        reader = tf.train.NewCheckpointReader(file_name)
        if not tensor_name:
            print(reader.debug_string().decode("utf-8"))
        else:
            print("tensor_name: ", tensor_name)
            print(reader.get_tensor(tensor_name))
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")


def main(unused_argv):
    if not FLAGS.file_name:
        print("Usage: inspect_checkpoint --file_name=checkpoint_file_name "
              "[--tensor_name=tensor_to_print]")
        sys.exit(1)
    else:
        print_tensors_in_checkpoint_file(FLAGS.file_name, FLAGS.tensor_name)


if __name__ == "__main__":
    tf.app.run()