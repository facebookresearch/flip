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

import configs.vit as vit
import configs.cfg_common_mae as cfg_common_mae


def get_config():
  """Get the hyperparameter configuration to train on TPUs."""
  config = cfg_common_mae.get_config()

  # mae config
  config.model.model_img.mask_ratio = 0.75
  config.model.model_img.norm_pix_loss = True

  config.model.model_img.update(vit.get_l16_config())
  config.model.model_img.hidden_size = 1280
  config.model.model_img.transformer.mlp_dim = config.model.model_img.hidden_size * 4
  config.model.model_img.transformer.dropout_rate = 0.0
  config.model.model_img.transformer.droppath_rate = 0.0
  config.model.model_img.transformer.num_layers = 32

  config.model.model_img.decoder = ml_collections.ConfigDict()
  config.model.model_img.decoder.hidden_size = 512
  config.model.model_img.decoder.transformer = ml_collections.ConfigDict()
  config.model.model_img.decoder.transformer.mlp_dim = config.model.model_img.decoder.hidden_size * 4
  config.model.model_img.decoder.transformer.num_heads = 16
  config.model.model_img.decoder.transformer.num_layers = 8
  config.model.model_img.decoder.transformer.attention_dropout_rate = 0.0
  config.model.model_img.decoder.transformer.dropout_rate = 0.0
  config.model.model_img.decoder.transformer.droppath_rate = 0.0

  # opt config
  config.opt_type = 'adarows'
  config.opt_mu_dtype = 'bfloat16'

  # save
  config.save_every_epochs = 10

  # vis
  # config.model.model_img.visualize = True

  return config
