# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import configs.default as default


def get_config():
    config = default.get_config()

    # 10000 imagenet epochs ~= 32 epoch LAION-400M
    config.num_epochs = 10000.0
    config.model.model_img.mask_ratio = 0.5

    # FLIP base
    config.model.model_img.hidden_size = 768
    config.model.model_img.transformer.num_heads = 12
    config.model.model_img.transformer.num_layers = 12

    config.model.model_txt.hidden_size = 512
    config.model.model_txt.transformer.num_heads = 8
    config.model.model_txt.transformer.num_layers = 12

    return config
