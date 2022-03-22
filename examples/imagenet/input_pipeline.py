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

from utils.img_transform_util import \
  decode_and_random_crop, \
  _decode_and_center_crop, normalize_image, color_jitter

from absl import logging
from PIL import Image
import io
from torchvision import transforms

from tensorflow.python.ops import random_ops
import torch

IMAGE_SIZE = 224


def get_torchvision_aug(image_size, aug):

  transform_aug = [
    transforms.RandomResizedCrop(image_size, scale=aug.area_range, ratio=aug.aspect_ratio_range, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip()]

  if aug.color_jit is not None:
    transform_aug += [transforms.ColorJitter(*aug.color_jit)]
          
  transform_aug += [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

  transform_aug = transforms.Compose(transform_aug)
  return transform_aug


def preprocess_for_train_torchvision(image_bytes, dtype=tf.float32, image_size=IMAGE_SIZE, transform_aug=None):
  """Preprocesses the given image for training.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    dtype: data type of the image.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = Image.open(io.BytesIO(image_bytes.numpy()))
  image = image.convert('RGB')

  # ------------------------------------------------------
  t_crop = transform_aug.transforms[0]
  t_flip = transform_aug.transforms[1]
  t_cjit = transform_aug.transforms[2]
  t_tten = transform_aug.transforms[3]
  t_norm = transform_aug.transforms[4]


  im = image
  im = t_crop(im)
  im = t_flip(im)
  im = t_tten(im)

  brightness = t_cjit.brightness
  factor = random_ops.random_uniform([], brightness[0], brightness[1]).numpy()
  # factor = torch.empty(1).uniform_(brightness[0], brightness[1]).numpy()
  im *= factor
  im = im.clip(min=0., max=1.)
  # im = t_cjit(im)

  im = t_norm(im)

  # im = t_cjit(image)
  # im = np.asarray(image)

  # import numpy as np
  # im = np.asarray(image)
  # transforms.ToTensor()(image)

  image = im
  # ------------------------------------------------------

  # image = transform_aug(image)
  image = tf.constant(image.numpy(), dtype=dtype)  # [3, 224, 224]
  image = tf.transpose(image, [1, 2, 0])  # [c, h, w] -> [h, w, c]
  return image


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
  image = tf.reshape(image, [image_size, image_size, 3])
  image = tf.image.random_flip_left_right(image)

  # advance augs
  if aug.color_jit is not None:
    image = color_jitter(image / 255., *aug.color_jit) * 255.  # color_jitter accept [0, 1] images

  image = normalize_image(image)
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
  if train:
    train_examples = dataset_builder.info.splits['train'].num_examples
    split_size = train_examples // jax.process_count()
    start = jax.process_index() * split_size
    split = 'train[{}:{}]'.format(start, start + split_size)
  else:
    validate_examples = dataset_builder.info.splits['validation'].num_examples
    split_size = validate_examples // jax.process_count()
    start = jax.process_index() * split_size
    split = 'validation[{}:{}]'.format(start, start + split_size)

  ds = dataset_builder.as_dataset(split=split, decoders={
      'image': tfds.decode.SkipDecoding(),
  })
  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  if aug is not None and aug.torchvision:
    options.experimental_threading.private_threadpool_size = 8
  ds = ds.with_options(options)

  if cache:
    ds = ds.cache()

  if train:
    ds = ds.repeat()
    ds = ds.shuffle(16 * batch_size, seed=0)

  if aug is not None and aug.torchvision:
    transform_aug = get_torchvision_aug(image_size, aug)
    logging.info(transform_aug)

    def decode_example(example):
      if train:
        image = preprocess_for_train_torchvision(example['image'], dtype, image_size, transform_aug=transform_aug)
      else:
        raise NotImplementedError
        image = preprocess_for_eval(example['image'], dtype, image_size)
      return {'image': image, 'label': example['label']}

    # ---------------------------------------
    # debugging torchvision's
    # x = next(iter(ds))
    # decode_example(x)
    # raise NotImplementedError
    # ---------------------------------------

    # kaiming: reference: https://github.com/tensorflow/tensorflow/issues/38212
    def py_func(image, label):
      d = decode_example({'image': image, 'label': label})
      return list(d.values())
    def ds_map_fn(x):
      flattened_output = tf.py_function(py_func, [x['image'], x['label']], [tf.float32, tf.int64])
      return {"image": flattened_output[0], "label": flattened_output[1]}

    ds = ds.map(ds_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  else:
    def decode_example(example):
      if train:
        image = preprocess_for_train(example['image'], dtype, image_size, aug=aug)
      else:
        image = preprocess_for_eval(example['image'], dtype, image_size)
      return {'image': image, 'label': example['label']}

    # ---------------------------------------
    # debugging tensorflow's
    # x = next(iter(ds))
    # decode_example(x)
    # raise NotImplementedError
    # ---------------------------------------


    ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)  

  ds = ds.batch(batch_size, drop_remainder=True)

  if not train:
    ds = ds.repeat()

  ds = ds.prefetch(10)

  return ds
