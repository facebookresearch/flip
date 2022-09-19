# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2021 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hyperparameter configuration to run the example on TPUs."""

import ml_collections

import configs.mae as mae


def get_config():
  """Get the hyperparameter configuration to train on TPUs."""
  config = ml_collections.ConfigDict()

  # As defined in the `models` module.
  # config.model = 'ResNet50'
  # `name` argument of tensorflow_datasets.builder()
  config.dataset = 'imagenet2012:5.*.*'

  config.learning_rate = 1.5e-4  # this is the base lr
  config.warmup_epochs = 40.0
  config.min_abs_lr = 0.  # this is abs lr
  config.warmup_abs_lr = 0.  # this is abs lr

  config.num_epochs = 100.0
  config.log_every_steps = 100
  config.save_every_epochs = 50

  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs using the entire dataset. Similarly for steps_per_eval.
  config.num_train_steps = -1
  config.steps_per_eval = -1

  # Consider setting the batch size to max(tpu_chips * 256, 8 * 1024) if you
  # train on a larger pod slice.
  config.batch_size = 4096
  config.cache = True

  # optimizer config
  config.opt_type = 'adamw'
  config.opt = ml_collections.ConfigDict()
  config.opt.b1 = 0.9
  config.opt.b2 = 0.95
  config.opt.weight_decay = 0.05
  
  config.opt_mu_dtype = 'float32'

  config.exclude_wd = True  # exclude some weight decays (bias, norm, cls, posembed)

  # aug config
  config.aug = ml_collections.ConfigDict()

  config.aug.area_range = (0.2, 1)
  config.aug.aspect_ratio_range = (3. / 4, 4. / 3.)
  config.aug.crop_ver = 'v4'  # v1, v3

  config.aug.label_smoothing = 0.0  # not used

  config.aug.autoaug = None  # autoaug, randaug, or None

  config.aug.color_jit = None  # [0.4, 0.4, 0.4]  # None to disable; [brightness, contrast, saturation]

  config.aug.eval_pad = 32  # ImageNet default eval: 224 crop from 256 (pad 32)

  # memory
  config.profile_memory = False

  # utils
  config.resume_dir = ''
  config.vis_every_epochs = 20.

  config.pretrain_dir = ''
  config.pretrain_fmt = 'jax'  # 't5x'

  # seeds
  config.seed_jax = 0
  config.seed_tf = 0
  config.seed_pt = 0

  # torchload
  config.torchload = ml_collections.ConfigDict()
  # config.torchload.data_dir = '/kmh_data/imagenet_full_size/061417'
  config.torchload.data_dir = '/datasets/imagenet-1k'
  config.torchload.num_workers = 32

  # partitioning
  config.partitioning = ml_collections.ConfigDict()
  config.partitioning.num_partitions = 1
  config.partitioning.partition_states = False

  # misc
  config.image_size = 224

  config.samples_per_epoch = 1281167  # define a "virtual" epoch

  # image+text model config
  config.model = mae.get_config()  # ViT-B/16

  # text aug config
  config.aug.txt = ml_collections.ConfigDict()
  config.aug.txt.tokenizer = 'tf_bert'  # 'tf_bert' 'hf_clip'
  config.aug.txt.max_len = 32  # clip: 77
  config.aug.txt.cls_token = True

  config.aug.txt.batch_process = False

  config.model.model_txt.vocab_size = 30523  # bert: 30522+1

  # clr config
  config.model.clr = ml_collections.ConfigDict()
  config.model.clr.tau = 0.2
  config.model.clr.tau_learnable = False
  config.model.clr.proj_layers = 2
  config.model.clr.proj_dim_hidden = 1024
  config.model.clr.proj_dim_out = 256

  config.model.clr.clr_loss = True

  # eval
  config.eval_only = False

  return config
