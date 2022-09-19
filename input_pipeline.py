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

import jax
import tensorflow as tf
import tensorflow_datasets as tfds

from utils.transform_util import \
  decode_and_random_crop, \
  _decode_and_center_crop, normalize_image, color_jitter

from utils.autoaug_util import distort_image_with_autoaugment, distort_image_with_randaugment
from utils import logging_util

from absl import logging
from PIL import Image
import io


def preprocess_for_train(image_bytes, dtype=tf.float32, image_size=None, aug=None):
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
  image = tf.reshape(image, [image_size, image_size, 3])
  image = tf.image.random_flip_left_right(image)

  # advance augs
  if aug.color_jit is not None:
    image = color_jitter(image / 255., *aug.color_jit) * 255.  # color_jitter accept [0, 1] images

  if aug.autoaug == 'autoaug':
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = tf.cast(image, dtype=tf.uint8)
    image = distort_image_with_autoaugment(image, 'v0')
    image = tf.cast(image, dtype=tf.float32)
  elif aug.autoaug == 'randaug':
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = tf.cast(image, dtype=tf.uint8)
    image = distort_image_with_randaugment(image, num_layers=2, magnitude=9)
    image = tf.cast(image, dtype=tf.float32)
  elif aug.autoaug is None:
    pass
  else:
    raise NotImplementedError

  image = normalize_image(image)
  image = tf.image.convert_image_dtype(image, dtype=dtype)
  return image


def preprocess_for_eval(image_bytes, dtype=tf.float32, image_size=None, aug=None):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    dtype: data type of the image.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_center_crop(image_bytes, image_size, pad=aug.eval_pad)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = normalize_image(image)
  image = tf.image.convert_image_dtype(image, dtype=dtype)
  return image


def create_split(dataset_builder, batch_size, data_layout, train, dtype=tf.float32,
                 image_size=None, cache=False, seed=0, aug=None):
  """Creates a split from the ImageNet dataset using TensorFlow Datasets.

  Args:
    dataset_builder: TFDS dataset builder for ImageNet.
    batch_size (local_batch_size): the batch size returned by the data pipeline.
    data_layout: the partitioner data_layout
    train: Whether to load the train or evaluation split.
    dtype: data type of the image.
    image_size: The target size of the images.
    cache: Whether to cache the dataset.
  Returns:
    A `tf.data.Dataset`.
  """
  shard_id = data_layout.shard_id
  num_shards = data_layout.num_shards

  if train:
    train_examples = dataset_builder.info.splits['train'].num_examples
    split_size = train_examples // num_shards
    start = shard_id * split_size
    split = 'train[{}:{}]'.format(start, start + split_size)
  else:
    validate_examples = dataset_builder.info.splits['validation'].num_examples
    split_size = validate_examples // num_shards
    start = shard_id * split_size
    split = 'validation[{}:{}]'.format(start, start + split_size)
  num_classes = dataset_builder.info.features['label'].num_classes

  ds = dataset_builder.as_dataset(split=split, decoders={
      'image': tfds.decode.SkipDecoding(),
  })
  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  if cache:
    ds = ds.cache()

  if train:  # force to shuffle
    ds = ds.repeat()
    ds = ds.shuffle(16 * batch_size, seed=seed)

  # define the decode function
  def decode_example(example):
    label = example['label']
    label_one_hot = tf.one_hot(label, depth=num_classes, dtype=dtype)
    if train:
      image = preprocess_for_train(example['image'], dtype, image_size, aug=aug)
      label_one_hot = label_one_hot * (1 - aug.label_smoothing) + aug.label_smoothing / num_classes
    else:
      image = preprocess_for_eval(example['image'], dtype, image_size, aug=aug)
    
    return {'image': image, 'label': label, 'label_one_hot': label_one_hot}

  # ---------------------------------------
  # debugging 
  # x = next(iter(ds))
  # decode_example(x)
  # raise NotImplementedError
  # ---------------------------------------

  ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(batch_size, drop_remainder=train)

  if not train:
    ds = ds.repeat()

  ds = ds.prefetch(10)

  return ds
