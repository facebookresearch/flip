import tensorflow as tf

from PIL import Image
import io
from torchvision import transforms

IMAGE_SIZE = 224
CROP_PADDING = 32


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


def preprocess_for_eval_torchvision(image_bytes, dtype=tf.float32, image_size=IMAGE_SIZE):
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

  transform_aug = [
    transforms.Resize(image_size + CROP_PADDING, interpolation=transforms.InterpolationMode.BICUBIC),  # 256
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
  transform_aug = transforms.Compose(transform_aug)

  image = transform_aug(image)
  image = tf.constant(image.numpy(), dtype=dtype)  # [3, 224, 224]
  image = tf.transpose(image, [1, 2, 0])  # [c, h, w] -> [h, w, c]
  return image