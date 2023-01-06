# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import ml_collections

import configs.default as default


def get_config():
    """
    default config for ViT-H/14 on LAION-2B with 20000 ImageNet epochs
    """
    config = default.get_config()

    config.num_epochs = 20000.0
    config.model.model_img.mask_ratio = 0.5
    config.warmup_epochs = 200

    # config.laion_path = LAION2B path
    config.learning_rate = 2e-6

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
