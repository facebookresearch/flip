# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import configs.default as default


def get_config():
    """
    default config for ViT-B on LAION-400M with 10000 ImageNet epochs
    """
    config = default.get_config()

    # config.laion_path = LAION-400M path
    # 10000 imagenet epochs ~= 32 epoch LAION-400M
    config.num_epochs = 10000.0
    config.model.model_img.mask_ratio = 0.5

    # FLIP base
    config.model.model_img.hidden_size = 768
    config.model.model_img.transformer.mlp_dim = config.model.model_img.hidden_size * 4
    config.model.model_img.transformer.num_heads = 12
    config.model.model_img.transformer.num_layers = 12

    config.model.model_txt.hidden_size = 512
    config.model.model_txt.transformer.mlp_dim = config.model.model_txt.hidden_size * 4
    config.model.model_txt.transformer.num_heads = 8
    config.model.model_txt.transformer.num_layers = 12

    config.model.clr.proj_dim_out = 512

    return config
