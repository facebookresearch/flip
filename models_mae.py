# Copyright 2021 Google LLC.
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

import functools
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as random

import flax.linen as nn

import t5x.layers


Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

# init hacks
# v1: JAX ViT; v2: PyTorch ViT; v3: v2 with fix
INIT_VER = 'v2'

fixed_gaussian_init = nn.initializers.normal(stddev=0.02)
if INIT_VER == 'v1':
  clstoken_init = nn.initializers.zeros
  posemb_init = fixed_gaussian_init
  patch_kernel_init = nn.initializers.lecun_uniform()
  patch_bias_init = nn.initializers.zeros
  msa_kernel_init = nn.initializers.xavier_uniform()
  mlp_kernel_init = nn.initializers.xavier_uniform()
  mlp_bias_init = nn.initializers.normal(stddev=1e-6)
  head_kernel_init=nn.initializers.zeros
elif INIT_VER == 'v2':
  clstoken_init = fixed_gaussian_init
  posemb_init = fixed_gaussian_init
  patch_kernel_init = fixed_gaussian_init
  patch_bias_init = fixed_gaussian_init  # bug from PyTorch code?
  msa_kernel_init = fixed_gaussian_init
  mlp_kernel_init = fixed_gaussian_init
  mlp_bias_init = nn.initializers.zeros
  # head_kernel_init = nn.initializers.normal(stddev=2e-5)
  head_kernel_init = fixed_gaussian_init
else:
  raise NotImplementedError


class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  @nn.compact
  def __call__(self, x):
    return x


# class AddPositionEmbs(nn.Module):
#   """Adds (optionally learned) positional embeddings to the inputs.

#   Attributes:
#     posemb_init: positional embedding initializer.
#   """

#   posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]

#   @nn.compact
#   def __call__(self, inputs):
#     """Applies AddPositionEmbs module.

#     By default this layer uses a fixed sinusoidal embedding table. If a
#     learned position embedding is desired, pass an initializer to
#     posemb_init.

#     Args:
#       inputs: Inputs to the layer.

#     Returns:
#       Output tensor with shape `(bs, timesteps, in_dim)`.
#     """
#     # inputs.shape is (batch_size, seq_len, emb_dim).
#     assert inputs.ndim == 3, ('Number of dimensions should be 3,'
#                               ' but it is: %d' % inputs.ndim)
#     pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
#     pe = t5x.layers.param_with_axes(
#         'pos_embedding',
#         self.posemb_init,
#         pos_emb_shape,
#         jnp.float32,
#         axes=('_null0', 'length', 'embed'))
#     return inputs + pe


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Attributes:
    posemb_init: positional embedding initializer.
  """
  sincos: bool
  use_cls_token: bool
  img_shape: Shape  # [h, w, c]
  dtype: Any = jnp.float32

  def setup(self):
    h, w, c = self.img_shape

    num_clstokens = 1 if self.use_cls_token else 0
    pos_emb_shape = (1, num_clstokens + h * w, c)  # (batch_size, seq_len, emb_dim).

    if not self.sincos:
      init_fn = posemb_init
      # self.pe = self.param('pos_embedding', posemb_init, pos_emb_shape)
    else:
      raise NotImplementedError
      pe_array = posembed_util.get_2d_sincos_pos_embed(c, (h, w), cls_token=self.use_cls_token)  # in numpy array

      init_fn = initializers_util.constant(value=pe_array, dtype=self.dtype)
      # self.pe = self.param('pos_embedding', sincos_init, pos_emb_shape)

    self.pe = t5x.layers.param_with_axes(
        'pos_embedding',
        init_fn,
        pos_emb_shape,
        jnp.float32,
        axes=('_null0', 'length', 'embed'))

    # kaiming: in MAE, we should always set posembed for cls_token as zero.
    # when loading for finetuning, this zero posembed can be tuned.
    # but this is not addressed here if sincos=False

  def __call__(self, inputs):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init.

    Args:
      inputs: Inputs to the layer.

    Returns:
      Output tensor with shape `(bs, timesteps, in_dim)`.
    """
    
    pe = jax.lax.stop_gradient(self.pe) if self.sincos else self.pe

    if self.use_cls_token:
      output = inputs + pe[:, 1:, :]
    else:
      output = inputs + pe

    return output


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: int
  dtype: Dtype = jnp.float32
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.xavier_uniform()
  bias_init: Callable[[PRNGKey, Shape, Dtype],
                      Array] = nn.initializers.normal(stddev=1e-6)

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = t5x.layers.Dense(
        features=self.mlp_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        kernel_axes=('embed', 'mlp'),
        name='Dense_0',
    )(inputs)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    x = t5x.layers.with_sharding_constraint(x, ('batch', 'length', 'mlp'))
    output = t5x.layers.Dense(
        features=actual_out_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        kernel_axes=('mlp', 'embed'),
        name='Dense_1',
    )(x)
    output = nn.Dropout(
        rate=self.dropout_rate)(
            output, deterministic=deterministic)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    inputs: input data.
    mlp_dim: dimension of the mlp on top of attention block.
    dtype: the dtype of the computation (default: float32).
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout for attention heads.
    deterministic: bool, deterministic or not (to apply dropout).
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
  """

  mlp_dim: int
  num_heads: int
  dtype: Dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  droppath_rate: float = 0.0
  layer_id: int = None

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Encoder1DBlock module.

    Args:
      inputs: Inputs to the layer.
      deterministic: Dropout will not be applied when set to true.

    Returns:
      output after transformer encoder block.
    """

    # Attention block.
    assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
    x = t5x.layers.LayerNorm(dtype=self.dtype, axes=('embed',))(inputs)

    # ----------------------------------------------------
    # t5x
    MsaBlock = functools.partial(
      t5x.layers.MultiHeadDotProductAttention,
      kernel_init=msa_kernel_init,
    )
    # original
    # MsaBlock = functools.partial(
    #   nn.MultiHeadDotProductAttention,
    #   kernel_init=msa_kernel_init,
    #   broadcast_dropout=False,
    #   deterministic=deterministic,
    # )
    # ----------------------------------------------------

    x = MsaBlock(
        dtype=self.dtype,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads,
    )(x, x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    # droppath
    x = nn.Dropout(rate=self.droppath_rate, broadcast_dims=(1, 2), name='droppath_msa')(x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = t5x.layers.LayerNorm(dtype=self.dtype, axes=('embed',))(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate,
        kernel_init=mlp_kernel_init,
        bias_init=mlp_bias_init,
        )(y, deterministic=deterministic)
    # droppath
    y = nn.Dropout(rate=self.droppath_rate, broadcast_dims=(1, 2), name='droppath_mlp')(y, deterministic=deterministic)

    return x + y


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Attributes:
    num_layers: number of layers
    mlp_dim: dimension of the mlp on top of attention block
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout rate in self attention.
  """

  num_layers: int
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  droppath_rate: float = 0.0

  @nn.compact
  def __call__(self, inputs, *, train, encoder_norm=True):
    """Applies Transformer model on the inputs.

    Args:
      inputs: Inputs to the layer.
      train: Set to `True` when training.

    Returns:
      output of a transformer encoder.
    """
    assert inputs.ndim == 3  # (batch, len, emb)

    x = inputs

    # Input Encoder
    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          droppath_rate=self.droppath_rate * lyr / (self.num_layers - 1),
          name='encoderblock_{:02d}'.format(lyr),
          num_heads=self.num_heads,
          layer_id=lyr,
        )(x, deterministic=not train)
    encoded = t5x.layers.LayerNorm(name='encoder_norm', axes=('embed',))(x) if encoder_norm else x

    return encoded


# the implemention for pmap
def gather(x, ids):
  return x[ids, :]
vmapped_gather = jax.vmap(gather, in_axes=(0, 0), out_axes=0, axis_name='batch')


# the implemention for pjit
def gather_by_einsum(x, ids):
  """kaiming: vmap + gather is slow with pjit; use einsum instead
  Args:
    x: [N, L, ...]
    ids: [N, K]
  """
  mat = jax.nn.one_hot(ids, x.shape[1])  # [N, K, L]
  x = jnp.einsum('nl...,nkl->nk...', x, mat)
  return x


class VisionTransformer(nn.Module):
  """VisionTransformer."""

  num_classes: int
  mask_ratio: float
  patches: Any
  transformer: Any
  hidden_size: int
  representation_size: Optional[int] = None
  classifier: str = 'token'
  dtype: Any = jnp.float32

  def random_mask(self, x):

    N, L, _ = x.shape  # batch, length, dim
    len_keep = int(L * (1 - self.mask_ratio))

    rng = self.make_rng('dropout')
    noise = random.uniform(rng, shape=x.shape[:2])

    ids_shuffle = jnp.argsort(noise, axis=1)  # ascend: small is keep, large is remove
    ids_restore = jnp.argsort(ids_shuffle, axis=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = gather_by_einsum(x, ids_keep)

    x_masked = t5x.layers.with_sharding_constraint(x_masked, ('batch', 'length', 'embed'))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = jnp.ones([N, L])
    mask = t5x.layers.with_sharding_constraint(mask, ('batch', 'length'))
    mask = mask.at[:, :len_keep].set(0)
    # unshuffle to get the binary mask
    mask = gather_by_einsum(mask, ids_restore)
    mask = t5x.layers.with_sharding_constraint(mask, ('batch', 'length'))

    return x_masked, mask, ids_restore

  def apply_encoder(self, inputs, train):
    use_cls_token=(self.classifier == 'token')
    assert use_cls_token  # kaiming: TODO: support both?

    x = t5x.layers.Conv(
        features=self.hidden_size,
        kernel_size=self.patches.size,
        strides=self.patches.size,
        padding='VALID',
        name='embedding',
        kernel_init=patch_kernel_init,
        bias_init=patch_bias_init,
        kernel_axes=('_null0', '_null1', '_null2', 'embed'),
        )(inputs)

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    x = AddPositionEmbs(sincos=False, use_cls_token=use_cls_token, img_shape=(h, w, c), name='posembed_encoder')(x)

    # masking: length -> length * mask_ratio
    x, mask, ids_restore = self.random_mask(x)
    ids_restore = jnp.reshape(ids_restore, [n, h, w])  # carries the shape info

    if use_cls_token:
      cls = t5x.layers.param_with_axes('cls', clstoken_init, (1, 1, c), jnp.float32, axes=('_null0', '_null1', 'embed'))
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    # apply the encoder
    x = Encoder(name='Transformer', **self.transformer)(x, train=train, encoder_norm=(self.classifier == 'token'))

    return x, mask, ids_restore


  @nn.compact
  def __call__(self, inputs, *, train):
    imgs = inputs

    # apply encoder
    x, mask, ids_restore = self.apply_encoder(imgs, train=train)

    if self.classifier == 'token':
      x = x[:, 0]
    elif self.classifier == 'tgap':
      x = x[:, 1:]
      x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))  # (1,) or (1,2)
      # x = nn.LayerNorm(name='fc_norm')(x)
      x = t5x.layers.LayerNorm(name='fc_norm', axes=('embed',))(x)
    elif self.classifier == 'gap':
      x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))  # (1,) or (1,2)
      # x = nn.LayerNorm(name='fc_norm')(x)
      x = t5x.layers.LayerNorm(name='fc_norm', axes=('embed',))(x)
    else:
      raise ValueError(f'Invalid classifier={self.classifier}')

    if self.representation_size is not None:
      raise NotImplementedError 
      # x = nn.Dense(features=self.representation_size, name='pre_logits')(x)
      # x = nn.tanh(x)
    else:
      x = IdentityLayer(name='pre_logits')(x)

    if self.num_classes:
      # x = nn.Dense(
      #   features=self.num_classes,
      #   name='head',
      #   kernel_init=head_kernel_init
      # )(x)      
      x = t5x.layers.Dense(
          features=self.num_classes,
          kernel_init=head_kernel_init,
          kernel_axes=('embed', 'classes'),
          name='head',
      )(x)

    return x