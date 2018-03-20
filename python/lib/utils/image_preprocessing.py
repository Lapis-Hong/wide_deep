#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/3/5
"""Provides custom function to preprocess images.
TODO: custom preprocess for CTR task
"""

import tensorflow as tf


def preprocess_image(image, is_training, height, width, depth):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_image_with_crop_or_pad(
            image, height + 8, width + 8)
        # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
        image = tf.random_crop(image, [height, width, depth])
        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    return image
