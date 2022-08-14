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

import os
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
# import tensorflow_text as tftx

from transformers import CLIPTokenizerFast

from utils.transform_util import \
  decode_and_random_crop, \
  _decode_and_center_crop, normalize_image, color_jitter

from utils.autoaug_util import distort_image_with_autoaugment, distort_image_with_randaugment
from utils import logging_util

from absl import logging
from PIL import Image
import io
import functools


feature_description = {
    'txt': tf.io.FixedLenFeature([], tf.string),
    'jpg': tf.io.FixedLenFeature([], tf.string),
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    # information related to downloading and processing
    'key': tf.io.FixedLenFeature([], tf.string),
    # 'original_height': tf.io.FixedLenFeature([], tf.int64),
    # 'original_width': tf.io.FixedLenFeature([], tf.int64),
    # 'NSFW': tf.io.FixedLenFeature([], tf.string),
    # 'url': tf.io.FixedLenFeature([], tf.string),
    # 'LICENSE': tf.io.FixedLenFeature([], tf.string),
    # 'similarity': tf.io.FixedLenFeature([], tf.float32),
}

def parse_laion_example(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)
    # example["image"] = tf.io.decode_image(example.pop("jpg"))
    example["image"] = example.pop("jpg")
    # the image id in the dataset (not continuous, due to download failures)
    example["image_id"] = tf.strings.to_number(example.pop("key"), out_type=tf.int64)
    return example


# define the decode function
def decode_example(example, image_size, aug, tokenize_func):
  # decoder the text
  txt = preprocess_text(example['txt'], tokenize_func=tokenize_func)

  # decoder the image
  image = preprocess_image(example['image'], image_size=image_size, aug=aug)
  return {'image': image, 'txt': txt}


def get_txt_tokenize_func(aug_txt):
  if aug_txt.tokenizer == 'clip':
    tokenizer_name = "openai/clip-vit-base-patch32"
    cache_dir = os.path.join("/kmh_data", tokenizer_name)  # point to the same location, in case it is changed
    tokenizer = CLIPTokenizerFast.from_pretrained(tokenizer_name, cache_dir=cache_dir)
  else:
    raise NotImplementedError

  tokenize = lambda s: tokenizer.encode(
    s.numpy().decode(),
    padding='max_length',
    truncation='longest_first',
    max_length=aug_txt.max_len,
    return_tensors='tf',)
  tokenize_tf_func = tf.function(
      lambda s: tf.py_function(tokenize, inp=[s], Tout=tf.int32),
      input_signature=[tf.TensorSpec(None, tf.string)],
  )
  return tokenize_tf_func


def preprocess_text(txt, tokenize_func):
  txt_enc = tokenize_func(txt)
  return txt_enc


def preprocess_image(image_bytes, dtype=tf.float32, image_size=None, aug=None):
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


def preprocess_image_for_eval(image_bytes, dtype=tf.float32, image_size=None):
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


def create_split(batch_size, data_layout, train, dtype=tf.float32,
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

  laion400m_dataset_path = "gs://kmh-gcp/laion-400m/tfrecord_dataset_img480"
  filenames = tf.io.gfile.glob(laion400m_dataset_path + "/*.tfrecord")  # len: 41408 (647*64)
  filenames.sort()

  if train:
    train_records = len(filenames)  # len: 41408
    split_size = train_records // num_shards
    start = shard_id * split_size
    split = 'train[{}:{}]'.format(start, start + split_size)

    # ----------------------------------------
    logging_util.verbose_on()
    logging_util.sync_and_delay()
    logging.info('Split: {} / {}'.format(split, train_records))
    logging_util.verbose_off()
    # ----------------------------------------

    filenames = filenames[start:start + split_size]
  else:
    raise NotImplementedError

  ds = tf.data.TFRecordDataset(filenames).map(parse_laion_example)
  ds = ds.apply(tf.data.experimental.ignore_errors(log_warning=False))

  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  if cache:
    ds = ds.cache()

  if train:
    ds = ds.repeat()
    ds = ds.shuffle(16 * batch_size, seed=seed)


  # create the tokenizer
  tokenize_func = get_txt_tokenize_func(aug.txt)
  decode_fn = functools.partial(decode_example, image_size=image_size, aug=aug, tokenize_func=tokenize_func)

  # ---------------------------------------
  # debugging 
  x = next(iter(ds))
  batch = decode_fn(x)
  # raise NotImplementedError
  # ---------------------------------------

  ds = ds.map(decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(batch_size, drop_remainder=True)

  if not train:
    ds = ds.repeat()

  ds = ds.prefetch(10)

  return ds