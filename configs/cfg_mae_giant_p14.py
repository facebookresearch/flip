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
  config.model.mask_ratio = 0.75
  config.model.norm_pix_loss = True

  config.model.update(vit.get_h14_config())
  config.model.patches = ml_collections.ConfigDict({'size': (14, 14)})
  config.model.hidden_size = 1664
  config.model.transformer.mlp_dim = 6656
  config.model.transformer.num_heads = 16
  config.model.transformer.num_layers = 48
  config.model.transformer.dropout_rate = 0.0
  config.model.transformer.droppath_rate = 0.0
  config.model.transformer.rescale_init = 1.0

  config.model.decoder = ml_collections.ConfigDict()
  config.model.decoder.hidden_size = 512
  config.model.decoder.transformer = ml_collections.ConfigDict()
  config.model.decoder.transformer.mlp_dim = config.model.decoder.hidden_size * 4
  config.model.decoder.transformer.num_heads = 16
  config.model.decoder.transformer.num_layers = 8
  config.model.decoder.transformer.attention_dropout_rate = 0.0
  config.model.decoder.transformer.dropout_rate = 0.0
  config.model.decoder.transformer.droppath_rate = 0.0

  # opt config
  config.opt_type = 'adamw'
  config.opt_mu_dtype = 'float32'

  # save
  config.save_every_epochs = 10

  # vis
  # config.model.visualize = True

  return config
