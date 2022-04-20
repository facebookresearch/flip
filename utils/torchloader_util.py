# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

import numpy as np
import jax

import torch
from torchvision import datasets, transforms

import timm
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.mixup import Mixup

from absl import logging

IMAGE_SIZE = 224

AUTOAUGS = {'autoaug': 'v0', 'randaugv2': 'rand-m9-mstd0.5-inc1'}


class ImageFolder(datasets.ImageFolder):
    """ImageFolder with label smoothing pre-process
    """
    def __init__(self, label_smoothing, **kwargs):
        super(ImageFolder, self).__init__(**kwargs)
        self.label_smoothing = label_smoothing
        self.num_classes = len(self.classes)

    def __getitem__(self, index: int):
        image, label = super(ImageFolder, self).__getitem__(index)
        label_one_hot = torch.nn.functional.one_hot(torch.tensor(label), self.num_classes).float()
        label_one_hot = label_one_hot * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
        
        # image = image.permute([1, 2, 0])  # chw2hwc

        return image, label, label_one_hot


def build_dataset(is_train, data_dir, aug):
    transform = build_transform(is_train, aug)
    label_smoothing = aug.label_smoothing if is_train else 0.

    root = os.path.join(data_dir, 'train' if is_train else 'val')
    dataset = ImageFolder(root=root, transform=transform, label_smoothing=label_smoothing)

    logging.info(dataset)

    return dataset


def build_transform(is_train, aug):
    input_size = IMAGE_SIZE

    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        color_jitter = 0.0 if aug.color_jit is None else aug.color_jit[0]
        aa = AUTOAUGS[aug.autoaug]
        re_prob = aug.randerase.prob if aug.randerase.on else 0.0
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=IMAGE_SIZE,
            is_training=True,
            color_jitter=color_jitter,
            auto_augment=aa,
            interpolation='bicubic',
            re_prob=re_prob,
            re_mode='pixel',
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def get_mixup_fn(aug, num_classes=1000):
    mixup_fn = None
    mixup_active = aug.mix.mixup or aug.mix.cutmix
    if mixup_active:
        logging.info("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=aug.mix.mixup_alpha,
            cutmix_alpha=aug.mix.cutmix_alpha,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode='batch',
            label_smoothing=aug.label_smoothing,
            num_classes=num_classes)
    return mixup_fn


def prepare_pt_data(xs, batch_size):
  """Convert a input batch from PyTorch Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x.numpy()  # pylint: disable=protected-access

    if x.shape[0] != batch_size:
      pads = -np.ones((batch_size - x.shape[0],) + x.shape[1:], dtype=x.dtype)
      x = np.concatenate([x, pads], axis=0)

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


# overwrite timm
def one_hot(x, num_classes, on_value=1., off_value=0., device='cpu'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)

def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cpu'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
    return y1 * lam + y2 * (1. - lam)

timm.data.mixup.one_hot = one_hot
timm.data.mixup.mixup_target = mixup_target
