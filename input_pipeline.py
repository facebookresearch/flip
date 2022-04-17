# Copyright 2022 The Flax Authors.
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

"""ImageNet input pipeline.
"""

import functools
from tensorflow.python.ops import random_ops

import jax
import tensorflow as tf
import tensorflow_datasets as tfds

from utils.transform_util import \
  decode_and_random_crop, \
  _decode_and_center_crop, normalize_image, color_jitter

from utils.autoaug_util import distort_image_with_autoaugment, distort_image_with_randaugment, distort_image_with_randaugment_v2
from utils.randerase_util import random_erase
from utils.torchvision_util import get_torchvision_aug, preprocess_for_train_torchvision, preprocess_for_eval_torchvision, get_torchvision_map_fn

from absl import logging

import math

IMAGE_SIZE = 224


def preprocess_for_train(image_bytes, dtype=tf.float32, image_size=IMAGE_SIZE, aug=None):
  """Preprocesses the given image for training.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    dtype: data type of the image.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
  crop_func = decode_and_random_crop[aug.crop_ver]
  image = crop_func(image_bytes, image_size, area_range=aug.area_range, aspect_ratio_range=aug.aspect_ratio_range)
  # image = tf.reshape(image, [image_size, image_size, 3])
  return image
  image = tf.image.random_flip_left_right(image)

  # advance augs
  if aug.color_jit is not None:
    image = color_jitter(image / 255., *aug.color_jit) * 255.  # color_jitter accept [0, 1] images

  autoaug_funtions = {
    'autoaug': functools.partial(distort_image_with_autoaugment, augmentation_name='v0'),
    'randaug': functools.partial(distort_image_with_randaugment, num_layers=2, magnitude=9),
    'randaugv2': functools.partial(distort_image_with_randaugment_v2, num_layers=2, magnitude=9),
  }

  if aug.autoaug in autoaug_funtions:
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = tf.cast(image, dtype=tf.uint8)
    image = autoaug_funtions[aug.autoaug](image)
    image = tf.cast(image, dtype=tf.float32)
  elif aug.autoaug is None or aug.autoaug == 'None':
    pass
  else:
    raise NotImplementedError

  image = normalize_image(image)

  # rand erase is after normalize
  if aug.randerase.on:
    image = random_erase(image, erase_prob=aug.randerase.prob)

  image = tf.image.convert_image_dtype(image, dtype=dtype)
  return image


def preprocess_for_eval(image_bytes, dtype=tf.float32, image_size=IMAGE_SIZE):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    dtype: data type of the image.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_center_crop(image_bytes, image_size)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = normalize_image(image)
  image = tf.image.convert_image_dtype(image, dtype=dtype)
  return image


def get_preprocess_for_train_func(image_size, aug, use_torchvision):
  if use_torchvision:
    transform_aug = get_torchvision_aug(image_size, aug)
    logging.info(transform_aug)
    return functools.partial(preprocess_for_train_torchvision, transform_aug=transform_aug)
  else:
    return functools.partial(preprocess_for_train, aug=aug)


def get_preprocess_for_eval_func(use_torchvision):
  if use_torchvision:
    return preprocess_for_eval_torchvision
  else:
    return preprocess_for_eval


def create_split(dataset_builder, batch_size, train, dtype=tf.float32,
                 image_size=IMAGE_SIZE, cache=False, aug=None):
  """Creates a split from the ImageNet dataset using TensorFlow Datasets.

  Args:
    dataset_builder: TFDS dataset builder for ImageNet.
    batch_size: the batch size returned by the data pipeline.
    train: Whether to load the train or evaluation split.
    dtype: data type of the image.
    image_size: The target size of the images.
    cache: Whether to cache the dataset.
  Returns:
    A `tf.data.Dataset`.
  """
  use_torchvision = (aug and aug.torchvision)

  if train:
    train_examples = dataset_builder.info.splits['train'].num_examples
    split_size = train_examples // jax.process_count()
    start = jax.process_index() * split_size
    split = 'train[{}:{}]'.format(start, start + split_size)
  else:
    validate_examples = dataset_builder.info.splits['validation'].num_examples
    split_size = math.ceil(validate_examples / jax.process_count())
    start = split_size * jax.process_index()
    end = min(start + split_size, validate_examples)
    split = 'validation[{}:{}]'.format(start, end)
    assert math.ceil(split_size / batch_size) == math.ceil((end - start) / batch_size)  # hack to make sure every host has the same # iter

  num_classes = dataset_builder.info.features['label'].num_classes

  ds = dataset_builder.as_dataset(split=split, decoders={
      'image': tfds.decode.SkipDecoding(),
  })
  options = tf.data.Options()
  options.threading.private_threadpool_size = 48 if not use_torchvision else 8
  options.deterministic = True
  ds = ds.with_options(options)

  if cache:
    ds = ds.cache()

  if train:
    ds = ds.repeat()
    # ds = ds.shuffle(512 * batch_size, seed=0)  # batch_size = 1024 (faster in local)
    ds = ds.shuffle(buffer_size=aug.shuffle_buffer_size, seed=0)

  preprocess_for_train_func = get_preprocess_for_train_func(image_size, aug, use_torchvision)
  preprocess_for_eval_func = get_preprocess_for_eval_func(use_torchvision)

  # define the decode function
  def decode_example(example):
    label = example['label']
    label_one_hot = tf.one_hot(label, depth=num_classes, dtype=dtype)
    if train:
      image = preprocess_for_train_func(example['image'], dtype, image_size)
      label_one_hot = label_one_hot * (1 - aug.label_smoothing) + aug.label_smoothing / num_classes
    else:
      image = preprocess_for_eval_func(example['image'], dtype, image_size)

    return {'image': image, 'label': label, 'label_one_hot': label_one_hot}

  if use_torchvision:
    ds_map_fn = get_torchvision_map_fn(decode_example)
  else:
    ds_map_fn = decode_example

  # ---------------------------------------
  # debugging 
  # x = next(iter(ds))
  # decode_example(x)
  # raise NotImplementedError
  # ---------------------------------------

  # ds = ds.map(ds_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(ds_map_fn, num_parallel_calls=1)

  ds = ds.batch(batch_size, drop_remainder=train)  # we drop the remainder if eval

  if not train:
    ds = ds.repeat()

  ds = ds.prefetch(10)

  return ds
