# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


"""Hyperparameter configuration."""

import ml_collections


def get_config():
    """Get the hyperparameter configuration to train on TPUs."""
    config = ml_collections.ConfigDict()

    # dataset
    config.eval_dataset = "imagenet2012:5.*.*"
    config.laion_path = ""

    config.learning_rate = 4e-6  # this is the base lr per 256 batch-size
    config.warmup_epochs = 20.0  # virutual imagenet epochs
    config.min_abs_lr = 0.0
    config.warmup_abs_lr = 0.0

    config.num_epochs = 10000.0  # virutual imagenet epochs
    config.log_every_steps = 200
    config.save_every_epochs = 100

    config.batch_size = 65536
    config.cache = True

    # optimizer config
    config.opt_type = "adamw"
    config.opt = ml_collections.ConfigDict()
    config.opt.b1 = 0.9
    config.opt.b2 = 0.95
    config.opt.weight_decay = 0.2
    config.opt_mu_dtype = "float32"
    config.exclude_wd = True  # exclude some weight decays (bias, norm, cls, posembed)
    config.opt.flatten_params = False

    # aug config
    config.aug = ml_collections.ConfigDict()
    config.aug.area_range = (0.5, 1)
    config.aug.aspect_ratio_range = (3.0 / 4, 4.0 / 3.0)
    config.aug.crop_ver = "v4"
    config.aug.eval_pad = 0
    config.aug.flip = True

    # text aug config
    config.aug.txt = ml_collections.ConfigDict()
    config.aug.txt.tokenizer = "tf_bert"
    config.aug.txt.max_len = 32
    config.aug.txt.cls_token = False

    # utils
    config.resume_dir = ""
    config.vis_every_epochs = 20.0
    config.pretrain_dir = ""

    # seeds
    config.seed_jax = 0
    config.seed_tf = 0
    config.seed_pt = 0

    # partitioning
    config.partitioning = ml_collections.ConfigDict()
    config.partitioning.num_partitions = 1
    config.partitioning.partition_states = False

    # misc
    config.image_size = 224
    config.samples_per_epoch = 1281167  # define a "virtual" epoch
    config.eval_only = False

    # flip model config
    config.model = ml_collections.ConfigDict()
    config.model.model_img = get_config_img()
    config.model.model_txt = get_config_txt()

    config.model.clr = ml_collections.ConfigDict()
    config.model.clr.tau = 0.01
    config.model.clr.tau_learnable = True
    config.model.clr.proj_dim_out = 768
    config.model.clr.proj_out_bias = False  # bias of the output proj layer
    config.model.clr.clr_loss = True
    config.model.clr.img_avg_token = True
    config.model.clr.txt_avg_token = False

    return config


def get_config_img():
    """Get the hyperparameter configuration for image model."""
    config = ml_collections.ConfigDict()

    config.mask_ratio = 0.5
    config.sincos = True

    config.name = "img_encoder"
    config.patches = ml_collections.ConfigDict({"size": (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = config.hidden_size * 4
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.transformer.droppath_rate = 0.0
    config.transformer.remat_policy = "none"

    return config


def get_config_txt():
    """Get the hyperparameter configuration for text model."""
    config = ml_collections.ConfigDict()

    config.mask_ratio = 0.0
    config.sincos = False

    config.name = "txt_encoder"
    config.vocab_size = 30523  # bert: 30522+1
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = config.hidden_size * 4
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.transformer.droppath_rate = 0.0

    return config
