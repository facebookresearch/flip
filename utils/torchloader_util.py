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

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from absl import logging

IMAGE_SIZE = 224


def build_dataset(is_train, config):
    transform = build_transform(is_train, config.aug)

    root = os.path.join(config.torchload.data_dir, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    logging.info(dataset)

    return dataset


def build_transform(is_train, aug):
    input_size = IMAGE_SIZE

    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        color_jitter = 0.0 if aug.color_jit is None else aug.color_jit[0]
        aa = {'autoaug': 'v0', 'randaugv2': 'rand-m9-mstd0.5-inc1'}
        aa = aa[aug.autoaug]
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
            re_count=1,
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
