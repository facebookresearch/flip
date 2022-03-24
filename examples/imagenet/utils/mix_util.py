import jax
import tensorflow_probability as tfp
import tensorflow as tf


def apply_mix(xs, cfg):
  """Apply mixup or cutmix to a batch.
  xs['image']: (N, H, W, C).
  xs['label']: (N, ). unchanged
  xs['label_one_hot']: (N, num_classes). had label smoothing.
  """
  imgs = xs['image']
  tgts = xs['label_one_hot']

  imgs_rev = tf.reverse(imgs, axis=[0])
  tgts_rev = tf.reverse(tgts, axis=[0])

  if cfg.mixup and cfg.cutmix and cfg.switch_elementwise:
    # element-wise mixup/cutmix switch (note lambda is always element-wise)
    use_mixup = (tf.random.uniform([imgs.shape[0]], minval=0, maxval=1., dtype=tf.float32) > 0.5)
    use_mixup = tf.cast(use_mixup, tf.float32)

    imgs_mixup, lmb_mixup = apply_mixup(imgs, imgs_rev, cfg.mixup_alpha)
    imgs_cutmix, lmb_cutmix = apply_cutmix(imgs, imgs_rev, cfg.cutmix_alpha)

    lmb = lmb_mixup * use_mixup + lmb_cutmix * (1 - use_mixup)

    use_mixup = tf.reshape(use_mixup, [-1] + [1] * (len(imgs.shape) - 1))
    imgs_mixed = imgs_mixup * use_mixup + imgs_cutmix * (1 - use_mixup)
  elif cfg.mixup and cfg.cutmix and not cfg.switch_elementwise:
    # host-wise mixup/cutmix switch (note lambda is always element-wise)
    use_mixup = (tf.random.uniform([], minval=0, maxval=1., dtype=tf.float32) > 0.5)
    imgs_mixed, lmb = tf.cond(use_mixup,
      lambda: apply_mixup(imgs, imgs_rev, cfg.mixup_alpha),
      lambda: apply_cutmix(imgs, imgs_rev, cfg.cutmix_alpha),
    )
  elif cfg.mixup and not cfg.cutmix:
    imgs_mixed, lmb = apply_mixup(imgs, imgs_rev, cfg.mixup_alpha)
  elif cfg.cutmix and not cfg.mixup:
    imgs_mixed, lmb = apply_cutmix(imgs, imgs_rev, cfg.cutmix_alpha)
  else:
    raise NotImplementedError

  # mix one-hot labels
  lmb = tf.reshape(lmb, [-1] + [1] * (len(tgts.shape) - 1))  # [N, 1]
  tgts_mixed = tgts * lmb + tgts_rev * (1. - lmb)

  return dict(image=imgs_mixed, label=xs['label'], label_one_hot=tgts_mixed)


def apply_mixup(imgs, imgs_rev, mixup_alpha):
  dist = tfp.distributions.Beta(mixup_alpha, mixup_alpha)
  lmb = dist.sample(imgs.shape[0])  # element-wise mix

  lmb_ = tf.reshape(lmb, [-1] + [1] * (len(imgs.shape) - 1))  # [N, 1, 1, 1]
  imgs_mixed = imgs * lmb_ + imgs_rev * (1 - lmb_)
  return imgs_mixed, lmb


def apply_cutmix(imgs, imgs_rev, cutmix_alpha):
  dist = tfp.distributions.Beta(cutmix_alpha, cutmix_alpha)
  lmb = dist.sample(imgs.shape[0])  # element-wise mix

  H, W = imgs.shape[1:3]
  assert H == W  # hack
  regions = get_cutmix_regions([H, W], target_area_ratio=1 - lmb)

  # correct lambda:
  lmb = 1. - tf.cast(regions[-1] * regions[-2], dtype=tf.float32) / float(H * W)

  # generate mask to perform images[:, oh:oh+h, ow:ow+w, :]
  masks = get_cutmix_masks([H, W], regions)  # 1 is remove, 0 is keep: tf.reduce_mean(masks, axis=[1,2,3]) == target_area_ratio (== 1-lmb)
  imgs_mixed = imgs * (1 - masks) + imgs_rev * masks

  return imgs_mixed, lmb


def get_cutmix_regions(img_size, target_area_ratio):
  """
  https://github.com/rwightman/pytorch-image-models/blob/02aaa785b97af5cbf22295033b4d3cc0137d8553/timm/data/mixup.py#L30
  target_area_ratio: a scalar or a 1-D tensor
  """
  img_h, img_w = img_size

  ratio = tf.sqrt(target_area_ratio)  # target_area_ratio is 1 - lambda

  cut_h = tf.cast(ratio * img_h, tf.int32)
  cut_w = tf.cast(ratio * img_w, tf.int32)

  cy = tf.random.uniform(target_area_ratio.shape, minval=0, maxval=img_h, dtype=tf.int32)
  cx = tf.random.uniform(target_area_ratio.shape, minval=0, maxval=img_w, dtype=tf.int32)

  yl = tf.clip_by_value(cy - cut_h // 2, 0, img_h)  # y_low
  yh = tf.clip_by_value(cy + cut_h // 2, 0, img_h)  # y_high

  xl = tf.clip_by_value(cx - cut_w // 2, 0, img_w)  # x_low
  xh = tf.clip_by_value(cx + cut_w // 2, 0, img_w)  # x_high

  # note: in timm, the box is accessed by: im[yl:yh, xl:xh]
  # in get_mask, the range is: mask_h = tf.math.logical_and((mask_h >= offset_h), (mask_h < offset_h + h))
  offset_h = yl
  offset_w = xl
  h = yh - yl
  w = xh - xl

  return [offset_h, offset_w, h, w]


def get_cutmix_masks(img_size, regions):
  """
  Generate cutmix mask to perform images[:, oh:oh+h, ow:ow+w, :]
  """
  H, W = img_size
  N = regions[0].shape.as_list()[0]
  offset_h, offset_w, h, w = regions  # each is (N,)
  
  mask_h = tf.range(H)  # (H,)
  mask_w = tf.range(W)  # (W,)

  mask_h = tf.repeat(tf.reshape(mask_h, [-1, 1]), N, axis=1)  # (H, N), int32, N in last for broadcast logical_and
  mask_w = tf.repeat(tf.reshape(mask_w, [-1, 1]), N, axis=1)  # (W, N), int32

  mask_h = tf.math.logical_and((mask_h >= offset_h), (mask_h < offset_h + h))  # (H, N)
  mask_w = tf.math.logical_and((mask_w >= offset_w), (mask_w < offset_w + w))  # (W, N)
  mask_h = tf.cast(mask_h, tf.float32)
  mask_w = tf.cast(mask_w, tf.float32)
  mask = tf.einsum('hn,wn->nhw', mask_h, mask_w)  # 1 is remove, 0 is keep: tf.reduce_mean(masks, axis=[1,2,3]) == target_area_ratio (== 1-lmb)
  mask = tf.reshape(mask, shape=[N, H, W, 1])
  return mask
