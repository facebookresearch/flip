# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import configs.default as default


def get_config():
    """
    default config for ViT-L on LAION-400M with 10000 ImageNet epochs
    """
    config = default.get_config()

    # config.laion_path = LAION-400M path
    # 10000 imagenet epochs ~= 32 epoch LAION-400M
    config.num_epochs = 10000.0
    config.model.model_img.mask_ratio = 0.5

    # save memory
    config.partitioning.partition_states = True

    return config
