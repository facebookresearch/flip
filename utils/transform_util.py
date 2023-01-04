# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import tensorflow as tf
from tensorflow.python.ops import random_ops

CROP_PADDING = 32
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


# crop v4: following PyTorch's implementation
def _decode_and_random_crop_v4(
    image_bytes,
    image_size,
    area_range=(0.08, 1.0),
    aspect_ratio_range=(3.0 / 4, 4.0 / 3.0),
):
    """
    reference:
    https://github.com/rwightman/pytorch-image-models/blob/ef72ad417709b5ba6404d85d3adafd830d507b2a/timm/data/transforms.py#L85
    """
    max_attempts = 10

    img_shape = tf.image.extract_jpeg_shape(image_bytes)
    area = tf.cast(img_shape[1] * img_shape[0], tf.float32)
    target_area = (
        tf.random.uniform(
            [max_attempts], area_range[0], area_range[1], dtype=tf.float32
        )
        * area
    )

    log_ratio = (tf.math.log(aspect_ratio_range[0]), tf.math.log(aspect_ratio_range[1]))
    aspect_ratio = tf.math.exp(
        tf.random.uniform([max_attempts], *log_ratio, dtype=tf.float32)
    )

    w = tf.cast(tf.round(tf.sqrt(target_area * aspect_ratio)), tf.int32)
    h = tf.cast(tf.round(tf.sqrt(target_area / aspect_ratio)), tf.int32)

    cond_w = w <= img_shape[1]
    cond_h = h <= img_shape[0]
    cond = tf.math.logical_and(cond_w, cond_h)

    # try to find the first True (if any)
    idx = tf.argsort(tf.cast(cond, tf.float32), direction="DESCENDING")[0]
    w, h, cond = w[idx], h[idx], cond[idx]

    # get the crop window: if valid (cond==True), use the rand crop; otherwise center crop
    crop_window = tf.cond(
        cond,
        lambda: _get_rand_crop_window(img_shape, w, h),
        lambda: _get_center_crop_window(img_shape, image_size),
    )

    image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    image = tf.image.resize(
        image, [image_size, image_size], tf.image.ResizeMethod.BICUBIC
    )  # not in TF MAE
    return image


def _get_rand_crop_window(img_size, w, h):
    # kaiming:
    # (i) in tf.random.uniform, the range is [minval, maxval), that is, maxval not included
    # (ii) in python's random.randint, the range is [minval, maxval]
    offset_w = tf.random.uniform(
        (), minval=0, maxval=img_size[1] - w + 1, dtype=tf.int32
    )
    offset_h = tf.random.uniform(
        (), minval=0, maxval=img_size[0] - h + 1, dtype=tf.int32
    )
    crop_window = tf.stack([offset_h, offset_w, h, w])
    return crop_window


def _get_center_crop_window(img_shape, image_size=224):
    # kaiming: copied from _decode_and_center_crop
    image_height = img_shape[0]
    image_width = img_shape[1]

    padded_center_crop_size = tf.cast(
        (
            (image_size / (image_size + CROP_PADDING))
            * tf.cast(tf.minimum(image_height, image_width), tf.float32)
        ),
        tf.int32,
    )

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack(
        [offset_height, offset_width, padded_center_crop_size, padded_center_crop_size]
    )
    return crop_window


def _decode_and_center_crop(image_bytes, image_size, pad, **kwargs):
    """Crops to center of image with padding then scales image_size."""
    shape = tf.io.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
        (
            (image_size / (image_size + pad))
            * tf.cast(tf.minimum(image_height, image_width), tf.float32)
        ),
        tf.int32,
    )

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack(
        [offset_height, offset_width, padded_center_crop_size, padded_center_crop_size]
    )
    image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    image = tf.compat.v1.image.resize_bicubic([image], [image_size, image_size])[
        0
    ]  # TF MAE: tf.image.resize_bicubic

    return image


decode_and_random_crop = {
    "v4": _decode_and_random_crop_v4,
    "vc": _decode_and_center_crop,
}


def normalize_image(image):
    image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
    return image
