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
from utils.randerase_util import random_erase

from absl import logging
from PIL import Image
import io
from torchvision import transforms


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

  image = transform_aug(image)
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
    end = start + split_size if jax.process_index() < jax.process_count() - 1 else validate_examples  # kaiming: this may fail if each host has different iterations
    split = 'validation[{}:{}]'.format(start, end)
    logging.set_verbosity(logging.INFO)  # show all processes
    logging.info('split: {}'.format(split))
    if not (jax.process_index() == 0):  # not first process
      logging.set_verbosity(logging.ERROR)  # disable info/warning

  num_classes = dataset_builder.info.features['label'].num_classes

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
    # ds = ds.shuffle(16 * batch_size, seed=0)
    ds = ds.shuffle(512 * batch_size, seed=0)  # batch_size = 1024 (faster in local)
  # else:
  #   assert len(ds) % 8 == 0  # we need the eval set to be divisible by 8 in this impl

  use_torchvision = (aug is not None and aug.torchvision)
  if use_torchvision:
    transform_aug = get_torchvision_aug(image_size, aug)
    logging.info(transform_aug)

  # define the decode function
  def decode_example(example):
    label = example['label']
    label_one_hot = tf.one_hot(label, depth=num_classes, dtype=dtype)
    if train:
      if use_torchvision:
        image = preprocess_for_train_torchvision(example['image'], dtype, image_size, transform_aug=transform_aug)
      else:
        image = preprocess_for_train(example['image'], dtype, image_size, aug=aug)
      label_one_hot = label_one_hot * (1 - aug.label_smoothing) + aug.label_smoothing / num_classes
    else:
      assert not use_torchvision
      image = preprocess_for_eval(example['image'], dtype, image_size)
    return {'image': image, 'label': label, 'label_one_hot': label_one_hot}

  if use_torchvision:
    # kaiming: reference: https://github.com/tensorflow/tensorflow/issues/38212
    def py_func(image, label):
      d = decode_example({'image': image, 'label': label})
      return list(d.values())
    def ds_map_fn(x):
      flattened_output = tf.py_function(py_func, [x['image'], x['label']], [tf.float32, tf.int64, tf.float32])
      return {"image": flattened_output[0], "label": flattened_output[1], "label_one_hot": flattened_output[2]}
  else:
    ds_map_fn = decode_example

  # ---------------------------------------
  # debugging 
  # x = next(iter(ds))
  # decode_example(x)
  # raise NotImplementedError
  # ---------------------------------------

  ds = ds.map(ds_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  ds = ds.batch(batch_size, drop_remainder=train)  # we drop the remainder if eval

  if not train:
    ds = ds.repeat()

  ds = ds.prefetch(10)

  return ds
