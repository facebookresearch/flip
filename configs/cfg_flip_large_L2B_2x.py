# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import configs.default as default


def get_config():
    """
    default config for ViT-L on LAION-2B with 20000 ImageNet epochs
    """
    config = default.get_config()

    config.num_epochs = 20000.0
    config.model.model_img.mask_ratio = 0.5

    # save memory
    config.partitioning.partition_states = True

    # config.laion_path = LAION2B path
    config.learning_rate = 2e-6

    return config
