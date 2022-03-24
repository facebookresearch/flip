import jax
import tensorflow_probability as tfp
import tensorflow as tf


def apply_mix(xs, aug):
  """Apply mixup or cutmix to a batch.
  xs['image']: (N, H, W, C).
  xs['label']: (N, ). unchanged
  xs['label_one_hot']: (N, num_classes). had label smoothing.
  """
  imgs = xs['image']
  tgts = xs['label_one_hot']

  dist = tfp.distributions.Beta(aug.mixup_alpha, aug.mixup_alpha)

  N = imgs.shape[0]
  lmb = dist.sample(N)  # element-wise mix

  # mixup
  lmb = tf.reshape(lmb, [N] + [1] * (len(imgs.shape) - 1))  # [N, 1, 1, 1]
  imgs_mixed = imgs * lmb + tf.reverse(imgs, axis=[0]) * (1. - lmb)
  lmb = tf.reshape(lmb, [N] + [1] * (len(tgts.shape) - 1))  # [N, 1]
  tgts_mixed = tgts * lmb + tf.reverse(tgts, axis=[0]) * (1. - lmb)

  return dict(image=imgs_mixed, label=xs['label'], label_one_hot=tgts_mixed)