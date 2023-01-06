# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import ml_collections

import configs.default as default


def get_config():
    """
    default config for ViT-H/14 on LAION-400M with 10000 ImageNet epochs
    """
    config = default.get_config()

    # config.laion_path = LAION-400M path
    # 10000 imagenet epochs ~= 32 epoch LAION-400M
    config.num_epochs = 10000.0
    config.model.model_img.mask_ratio = 0.5
    config.warmup_epochs = 200

    # FLIP huge
    config.model.model_img.patches = ml_collections.ConfigDict({"size": (14, 14)})
    config.model.model_img.hidden_size = 1280
    config.model.model_img.transformer.mlp_dim = config.model.model_img.hidden_size * 4
    config.model.model_img.transformer.num_heads = 16
    config.model.model_img.transformer.num_layers = 32

    config.model.model_txt.hidden_size = 1024
    config.model.model_txt.transformer.mlp_dim = config.model.model_txt.hidden_size * 4
    config.model.model_txt.transformer.num_heads = 16
    config.model.model_txt.transformer.num_layers = 24

    config.model.clr.proj_dim_out = 1024

    # save memory
    config.partitioning.partition_states = True
    # flatten params in optmizer
    config.opt.flatten_params = True

    return config
