# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import configs.default as default


def get_config():
    """
    default config for ViT-L on LAION-2B with 20000 ImageNet epochs
    """
    config = default.get_config()

    config.num_epochs = 20000.0
    config.model.model_img.mask_ratio = 0.5

    # config.laion_path = LAION2B path
    config.learning_rate = 2e-6

    return config