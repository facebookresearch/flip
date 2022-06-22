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

# for visualization
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


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
        try:
            image, label = super(ImageFolder, self).__getitem__(index)
        except Exception as e:
            logging.info('Replacing image due to error: {}'.format(e))
            image, label = super(ImageFolder, self).__getitem__((index + 1) % self.__len__())  # offset
        label_one_hot = torch.nn.functional.one_hot(torch.tensor(label), self.num_classes).float()
        label_one_hot = label_one_hot * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes

        return image, label, label_one_hot

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        body.append("Number of classes: {}".format(self.num_classes))
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        if self.target_transform is not None:
            body.append("Target transform: {}".format(self.target_transform))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)


def build_dataset(is_train, data_dir, aug):
    transform = build_transform(is_train, aug)
    label_smoothing = aug.label_smoothing if is_train else 0.

    if 'imagenet-1k' in data_dir or 'imagenet_full_size/061417' in data_dir:  # IN-1K
        root = os.path.join(data_dir, 'train' if is_train else 'val')
    elif 'imagenet-22k' in data_dir:  # IN-22K
        root = os.path.join(data_dir, '062717') if is_train \
            else '/datasets/imagenet-1k/val'
    else:
        raise NotImplementedError

    dataset = ImageFolder(root=root, transform=transform, label_smoothing=label_smoothing)

    logging.info(dataset)

    return dataset


def build_transform(is_train, aug):
    input_size = IMAGE_SIZE

    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        color_jitter = None if aug.color_jit is None else aug.color_jit[0]
        aa = AUTOAUGS[aug.autoaug] if aug.autoaug else None
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=IMAGE_SIZE,
            is_training=True,
            scale=aug.area_range,
            ratio=aug.aspect_ratio_range,
            color_jitter=color_jitter,
            auto_augment=aa,
            interpolation='bicubic',
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


def get_target_transform_1k_to_22k(dataset_train, dataset_val):
    class_to_idx_val = dataset_val.class_to_idx
    class_to_idx_train = dataset_train.class_to_idx
  
    idx_val_to_train = {}
    for k, v in class_to_idx_val.items():
        if k == 'n04399382':
            idx_val_to_train[v] = 0 # the only non-overlapping case; assign a random number
        else:
            idx_val_to_train[v] = class_to_idx_train[k]
    
    def target_transform(target):
        return idx_val_to_train[target]
    
    return target_transform
