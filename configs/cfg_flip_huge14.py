# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import ml_collections

import configs.default as default


def get_config():
    config = default.get_config()

    # 10000 imagenet epochs ~= 32 epoch LAION-400M
    config.num_epochs = 10000.0
    config.model.model_img.mask_ratio = 0.5
    config.warmup_epochs = 200

    # FLIP huge
    config.model.model_img.patches = ml_collections.ConfigDict({"size": (14, 14)})
    config.model.model_img.hidden_size = 1280
    config.model.model_img.transformer.num_heads = 16
    config.model.model_img.transformer.num_layers = 32

    config.model.model_txt.hidden_size = 1024
    config.model.model_txt.transformer.num_heads = 16
    config.model.model_txt.transformer.num_layers = 24

    return config
