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
import configs.cfg_common_ft as cfg_common_ft


def get_config():
  """Get the hyperparameter configuration to train on TPUs."""
  config = cfg_common_ft.get_config()

  # model config
  config.model.update(vit.get_testing_config())
  config.model.transformer.dropout_rate = 0.0
  config.model.transformer.droppath_rate = 0.1
  config.model.transformer.num_layers = 1

  config.model.hidden_size = 1024
  config.model.transformer.mlp_dim = 4096
  config.model.transformer.num_heads = 16

  config.model.classifier = 'tgap'  # 'token', 'tgap' (token + gap)

  # opt config
  config.opt_mu_dtype = 'float32'

  return config
