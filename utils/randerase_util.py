# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Random Erase util
Reference: https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/data/random_erasing.py#L68
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utils.transform_util import _get_rand_crop_window


def random_erase(image, erase_prob=0.25):
  """Apply RandErase with a probability
  image: (H, W, C)
  """
  apply_erase = (tf.random.uniform(shape=(), minval=0., maxval=1.0) < erase_prob)  # bool
  image = tf.cond(apply_erase,
    lambda: _apply_random_erase(image),
    lambda: image
  )
  return image


def _apply_random_erase(image):
  """Apply RandErase
  image: (H, W, C)
  """
  H, W, C = image.shape

  # get a random region
  region = _get_random_region([H, W], area_range=(0.02, 1 / 3), aspect_ratio_range=(0.3, 1 / 0.3))

  # turn the region into a mask
  mask = _get_mask([H, W], region)  # [h, w, 1], 1 is region, 0 is outside: tf.reduce_mean(mask) = target_area_ratio

  # generate the noise
  noise = tf.random.normal([H, W, C], dtype=tf.float32)
  noise = noise * 100000

  # mix with the mask
  image = image * (1 - mask) + noise * mask

  return image


def _get_random_region(img_shape, area_range, aspect_ratio_range):
  """
  reference:
  https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/data/random_erasing.py#L68
  kaiming: implemented like crop_v4
  """
  max_attempts = 10

  area = tf.cast(img_shape[1] * img_shape[0], tf.float32)
  target_area = tf.random.uniform([max_attempts], area_range[0], area_range[1], dtype=tf.float32) * area

  log_ratio = (tf.math.log(aspect_ratio_range[0]), tf.math.log(aspect_ratio_range[1]))
  aspect_ratio = tf.math.exp(
      tf.random.uniform([max_attempts], *log_ratio, dtype=tf.float32))

  h = tf.cast(tf.round(tf.sqrt(target_area * aspect_ratio)), tf.int32)  # note: different from rand crop
  w = tf.cast(tf.round(tf.sqrt(target_area / aspect_ratio)), tf.int32)

  cond_w = (w <= img_shape[1])
  cond_h = (h <= img_shape[0])
  cond = tf.math.logical_and(cond_w, cond_h)

  # try to find the first True (if any)
  idx = tf.argsort(tf.cast(cond, tf.float32), direction='DESCENDING')[0]
  w, h, cond = w[idx], h[idx], cond[idx]

  # get the crop window: if valid (cond==True), use the rand crop; otherwise center crop
  crop_window = tf.cond(cond,
    lambda: _get_rand_crop_window(img_shape, w, h),
    lambda: _get_rand_crop_window(img_shape, 0, 0)
  )
  return crop_window


def _get_mask(img_size, region):
  H, W = img_size 
  offset_h, offset_w, h, w = region
  mask_h = tf.range(H)
  mask_w = tf.range(W)
  mask_h = tf.math.logical_and((mask_h >= offset_h), (mask_h < offset_h + h))
  mask_w = tf.math.logical_and((mask_w >= offset_w), (mask_w < offset_w + w))
  mask_h = tf.cast(mask_h, tf.float32)
  mask_w = tf.cast(mask_w, tf.float32)
  mask = tf.einsum('h,w->hw', mask_h, mask_w)  # 1 is region, 0 is outside: tf.reduce_mean(mask) = target_area_ratio
  mask = tf.reshape(mask, shape=[H, W, 1])
  return mask