# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
# https://github.com/google/flax/tree/main/examples/imagenet


import tensorflow as tf
import tensorflow_datasets as tfds

from utils.transform_util import (
    _decode_and_center_crop,
    normalize_image,
)


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


def create_split(
    dataset_builder,
    batch_size,
    data_layout,
    train,
    dtype=tf.float32,
    image_size=None,
    cache=False,
    seed=0,
    aug=None,
):
    """Creates a split from the ImageNet dataset using TensorFlow Datasets.

    Args:
        dataset_builder: TFDS dataset builder for ImageNet.
        batch_size (local_batch_size): the batch size returned by the data pipeline.
        data_layout: the partitioner data_layout
        train: Whether to load the train or evaluation split.
        dtype: data type of the image.
        image_size: The target size of the images.
        cache: Whether to cache the dataset.
                seed: seed.
        aug: config for augmentations.
    Returns:
        A `tf.data.Dataset`.
    """
    shard_id = data_layout.shard_id
    num_shards = data_layout.num_shards

    assert train == False, "FLIP training only needs ImageNet for evaluation"

    validate_examples = dataset_builder.info.splits["validation"].num_examples
    split_size = validate_examples // num_shards
    start = shard_id * split_size
    split = "validation[{}:{}]".format(start, start + split_size)
    num_classes = dataset_builder.info.features["label"].num_classes

    ds = dataset_builder.as_dataset(
        split=split,
        decoders={
            "image": tfds.decode.SkipDecoding(),
        },
    )
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    ds = ds.with_options(options)

    if cache:
        ds = ds.cache()

    if train:
        ds = ds.repeat()
        ds = ds.shuffle(16 * batch_size, seed=seed)

    # define the decode function
    def decode_example(example):
        label = example["label"]
        label_one_hot = tf.one_hot(label, depth=num_classes, dtype=dtype)
        image = preprocess_for_eval(example["image"], dtype, image_size, aug=aug)

        return {"image": image, "label": label, "label_one_hot": label_one_hot}

    ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=train)

    if not train:
        ds = ds.repeat()

    ds = ds.prefetch(10)

    return ds
