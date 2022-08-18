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


def get_config():
  """Get the hyperparameter configuration to train on TPUs."""
  config = ml_collections.ConfigDict()
  config.model_img = get_config_img()
  config.model_txt = get_config_txt()

  config.visualize = True

  return config


def get_config_img():
  """Get the hyperparameter configuration to train on TPUs."""
  config = ml_collections.ConfigDict()

  config.mask_ratio = 0.75
  config.norm_pix_loss = True

  config.sincos = True

  config.name = 'img_encoder'
  config.patches = ml_collections.ConfigDict({'size': (16, 16)})
  config.hidden_size = 768
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 3072
  config.transformer.num_heads = 12
  config.transformer.num_layers = 12
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.transformer.droppath_rate = 0.0
  config.classifier = 'token'

  config.decoder = ml_collections.ConfigDict()
  config.decoder.name = 'img_decoder'
  config.decoder.hidden_size = 256
  config.decoder.transformer = ml_collections.ConfigDict()
  config.decoder.transformer.mlp_dim = config.hidden_size * 4
  config.decoder.transformer.num_heads = 16
  config.decoder.transformer.num_layers = 4
  config.decoder.transformer.attention_dropout_rate = 0.0
  config.decoder.transformer.dropout_rate = 0.0
  config.decoder.transformer.droppath_rate = 0.0

  config.decoder.cross_attention = True

  config.decoder.on_use = True  # whehter img has decoders?

  return config


def get_config_txt():
  """Get the hyperparameter configuration to train on TPUs."""
  config = ml_collections.ConfigDict()

  config.mask_ratio = 0.25

  config.sincos = False

  config.name = 'txt_encoder'
  config.vocab_size = 0
  config.hidden_size = 768
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = config.hidden_size * 4
  config.transformer.num_heads = 12
  config.transformer.num_layers = 12
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.transformer.droppath_rate = 0.0

  config.decoder = ml_collections.ConfigDict()
  config.decoder.name = 'txt_decoder'
  config.decoder.hidden_size = 256
  config.decoder.transformer = ml_collections.ConfigDict()
  config.decoder.transformer.mlp_dim = config.hidden_size * 4
  config.decoder.transformer.num_heads = 16
  config.decoder.transformer.num_layers = 4
  config.decoder.transformer.attention_dropout_rate = 0.0
  config.decoder.transformer.dropout_rate = 0.0
  config.decoder.transformer.droppath_rate = 0.0

  config.decoder.cross_attention = True

  config.decoder.on_use = True  # whehter txt has decoders?

  return config
