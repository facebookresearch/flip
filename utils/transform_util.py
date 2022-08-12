import tensorflow as tf
from tensorflow.python.ops import random_ops

from absl import logging

CROP_PADDING = 32
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def color_jitter(image,
                 brightness=0, contrast=0, saturation=0,
                 random_order=True):
  """Distorts the color of the image.

  Args:
    image: The input image tensor.
    strength: the floating number for the strength of the color augmentation.
    random_order: A bool, specifying whether to randomize the jittering order.

  Returns:
    The distorted image tensor.
  """
  hue = 0.0
  if random_order:
    return color_jitter_rand(image, brightness, contrast, saturation, hue)
  else:
    return color_jitter_nonrand(image, brightness, contrast, saturation, hue)


def random_brightness(x, max_delta):
  """Multiplicative brightness jitter"""
  delta = random_ops.random_uniform([], -max_delta, max_delta)
  x = x * (1. + delta)
  return x


def color_jitter_nonrand(image, brightness=0, contrast=0, saturation=0, hue=0):
  """Distorts the color of the image (jittering order is fixed).

  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.

  Returns:
    The distorted image tensor.
  """
  with tf.name_scope('distort_color'):
    def apply_transform(i, x, brightness, contrast, saturation, hue):
      """Apply the i-th transformation."""
      if brightness != 0 and i == 0:
        # x = tf.image.random_brightness(x, max_delta=brightness)
        return random_brightness(x, max_delta=brightness)
      elif contrast != 0 and i == 1:
        x = tf.image.random_contrast(
            x, lower=1-contrast, upper=1+contrast)
      elif saturation != 0 and i == 2:
        x = tf.image.random_saturation(
            x, lower=1-saturation, upper=1+saturation)
      elif hue != 0:
        x = tf.image.random_hue(x, max_delta=hue)
      return x

    for i in range(4):
      image = apply_transform(i, image, brightness, contrast, saturation, hue)
      image = tf.clip_by_value(image, 0., 1.)
    return image


def color_jitter_rand(image, brightness=0, contrast=0, saturation=0, hue=0):
  """Distorts the color of the image (jittering order is random).

  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.

  Returns:
    The distorted image tensor.
  """
  with tf.name_scope('distort_color'):
    def apply_transform(i, x):
      """Apply the i-th transformation."""
      def brightness_foo():
        if brightness == 0:
          return x
        else:
          # return tf.image.random_brightness(x, max_delta=brightness)  # this is additive 
          return random_brightness(x, max_delta=brightness)
      def contrast_foo():
        if contrast == 0:
          return x
        else:
          return tf.image.random_contrast(x, lower=1-contrast, upper=1+contrast)
      def saturation_foo():
        if saturation == 0:
          return x
        else:
          return tf.image.random_saturation(
              x, lower=1-saturation, upper=1+saturation)
      def hue_foo():
        if hue == 0:
          return x
        else:
          return tf.image.random_hue(x, max_delta=hue)
      x = tf.cond(tf.less(i, 2),
                  lambda: tf.cond(tf.less(i, 1), brightness_foo, contrast_foo),
                  lambda: tf.cond(tf.less(i, 3), saturation_foo, hue_foo))
      return x

    perm = tf.random.shuffle(tf.range(4))
    for i in range(4):
      image = apply_transform(perm[i], image)
      image = tf.clip_by_value(image, 0., 1.)
    return image


def distorted_bounding_box_crop(image_bytes,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100):
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image_bytes: `Tensor` of binary image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
        where each coordinate is [0, 1) and the coordinates are arranged
        as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding
        box supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
  Returns:
    cropped image `Tensor`
  """
  shape = tf.io.extract_jpeg_shape(image_bytes)
  sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
      shape,
      bounding_boxes=bbox,
      min_object_covered=min_object_covered,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=max_attempts,
      use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, _ = sample_distorted_bounding_box

  # Crop the image to the specified bounding box.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
  image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

  return image


def _at_least_x_are_equal(a, b, x):
  """At least `x` of `a` and `b` `Tensors` are equal."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


# crop_v1: TF default 
def _decode_and_random_crop(image_bytes, image_size,
    area_range=(0.08, 1.0), aspect_ratio_range=(3. / 4, 4. / 3.)):
  """Make a random crop of image_size.
  """
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = distorted_bounding_box_crop(
      image_bytes,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=10)
  original_shape = tf.io.extract_jpeg_shape(image_bytes)
  bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

  image = tf.cond(
      bad,
      lambda: _decode_and_center_crop(image_bytes, image_size),
      lambda: tf.compat.v1.image.resize_bicubic([image], [image_size, image_size])[0]  # TF MAE: tf.image.resize_bicubic
  )

  return image


# crop v2: hand-written, following BYOL
def _decode_and_random_crop_v2(image_bytes, image_size,
    area_range=(0.08, 1.0), aspect_ratio_range=(3. / 4, 4. / 3.)):

  img_size = tf.image.extract_jpeg_shape(image_bytes)
  area = tf.cast(img_size[1] * img_size[0], tf.float32)
  target_area = tf.random.uniform([], area_range[0], area_range[1], dtype=tf.float32) * area

  log_ratio = (tf.math.log(aspect_ratio_range[0]), tf.math.log(aspect_ratio_range[1]))
  aspect_ratio = tf.math.exp(
      tf.random.uniform([], *log_ratio, dtype=tf.float32))

  w = tf.cast(tf.round(tf.sqrt(target_area * aspect_ratio)), tf.int32)
  h = tf.cast(tf.round(tf.sqrt(target_area / aspect_ratio)), tf.int32)

  w = tf.minimum(w, img_size[1])
  h = tf.minimum(h, img_size[0])

  offset_w = tf.random.uniform((),
                              minval=0,
                              maxval=img_size[1] - w + 1,
                              dtype=tf.int32)
  offset_h = tf.random.uniform((),
                              minval=0,
                              maxval=img_size[0] - h + 1,
                              dtype=tf.int32)

  crop_window = tf.stack([offset_h, offset_w, h, w])
  image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  # image = tf.image.resize(image, [image_size, image_size], tf.image.ResizeMethod.BICUBIC)  # TF MAE: tf.image.resize (different than crop v1, v3)
  logging.warn('Using tf.compat.v1.image.resize_bicubic. This is altered from TF MAE API.')
  image =  tf.compat.v1.image.resize_bicubic([image], [image_size, image_size])[0]
  return image


# crop v3: like SimCLR's original code, but: (i) max_attempts=100, and (ii) remove bad condition
def _decode_and_random_crop_v3(image_bytes, image_size,
    area_range=(0.08, 1.0), aspect_ratio_range=(3. / 4, 4. / 3.)):
  """Make a random crop of image_size."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = distorted_bounding_box_crop(
      image_bytes,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=100)
  image =  tf.compat.v1.image.resize_bicubic([image], [image_size, image_size])[0] # TF MAE: tf.image.resize_bicubic

  return image


# crop v4: another attempt of following PyTorch's implementation
def _decode_and_random_crop_v4(image_bytes, image_size,
    area_range=(0.08, 1.0), aspect_ratio_range=(3. / 4, 4. / 3.)):
  """
  reference:
  https://github.com/rwightman/pytorch-image-models/blob/ef72ad417709b5ba6404d85d3adafd830d507b2a/timm/data/transforms.py#L85
  """
  max_attempts = 10

  img_shape = tf.image.extract_jpeg_shape(image_bytes)
  area = tf.cast(img_shape[1] * img_shape[0], tf.float32)
  target_area = tf.random.uniform([max_attempts], area_range[0], area_range[1], dtype=tf.float32) * area

  log_ratio = (tf.math.log(aspect_ratio_range[0]), tf.math.log(aspect_ratio_range[1]))
  aspect_ratio = tf.math.exp(
      tf.random.uniform([max_attempts], *log_ratio, dtype=tf.float32))

  w = tf.cast(tf.round(tf.sqrt(target_area * aspect_ratio)), tf.int32)
  h = tf.cast(tf.round(tf.sqrt(target_area / aspect_ratio)), tf.int32)

  cond_w = (w <= img_shape[1])
  cond_h = (h <= img_shape[0])
  cond = tf.math.logical_and(cond_w, cond_h)

  # try to find the first True (if any)
  idx = tf.argsort(tf.cast(cond, tf.float32), direction='DESCENDING')[0]
  w, h, cond = w[idx], h[idx], cond[idx]

  # get the crop window: if valid (cond==True), use the rand crop; otherwise center crop
  crop_window = tf.cond(cond,
    lambda: _get_rand_crop_window(img_shape, w, h),
    lambda: _get_center_crop_window(img_shape, image_size)
  )
  
  image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  image = tf.image.resize(image, [image_size, image_size], tf.image.ResizeMethod.BICUBIC)  # not in TF MAE
  return image


def _get_rand_crop_window(img_size, w, h):
  # kaiming:
  # (i) in tf.random.uniform, the range is [minval, maxval), that is, maxval not included
  # (ii) in python's random.randint, the range is [minval, maxval]
  offset_w = tf.random.uniform((),
                              minval=0,
                              maxval=img_size[1] - w + 1,
                              dtype=tf.int32)
  offset_h = tf.random.uniform((),
                              minval=0,
                              maxval=img_size[0] - h + 1,
                              dtype=tf.int32)
  crop_window = tf.stack([offset_h, offset_w, h, w])
  return crop_window


def _get_center_crop_window(img_shape, image_size=224):
  # kaiming: copied from _decode_and_center_crop
  image_height = img_shape[0]
  image_width = img_shape[1]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + CROP_PADDING)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)),
      tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([offset_height, offset_width,
                          padded_center_crop_size, padded_center_crop_size])
  return crop_window
  

def _decode_and_center_crop(image_bytes, image_size, **kwargs):
  """Crops to center of image with padding then scales image_size."""
  shape = tf.io.extract_jpeg_shape(image_bytes)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + CROP_PADDING)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)),
      tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([offset_height, offset_width,
                          padded_center_crop_size, padded_center_crop_size])
  image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  image = tf.compat.v1.image.resize_bicubic([image], [image_size, image_size])[0]  # TF MAE: tf.image.resize_bicubic

  return image


decode_and_random_crop ={
    'v1': _decode_and_random_crop,
    'v2': _decode_and_random_crop_v2,
    'v3': _decode_and_random_crop_v3,
    'v4': _decode_and_random_crop_v4,
    'vc': _decode_and_center_crop,
  }


def normalize_image(image):
  image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
  image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
  return image
