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
from xml.sax.xmlreader import InputSource

from absl import logging
import math

import jax
import jax.numpy as jnp
import jax.random as random
import optax

import flax.linen as nn
from flax.linen.partitioning import remat

import t5x.layers

from utils import posembed_util
from utils import initializers_util
from utils.memory_queue import MemQueue


Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

# init hacks
INIT_VER = 'mae_jax_v2'

fixed_gaussian_init = nn.initializers.normal(stddev=0.02)
if INIT_VER == 'mae_jax_v2':
  clstoken_init = fixed_gaussian_init
  masktoken_init = fixed_gaussian_init
  posemb_init = fixed_gaussian_init  # not used if sincos
  
  # patch_kernel_init = fixed_gaussian_init
  patch_kernel_init = initializers_util.patch_kernel()
  patch_bias_init = nn.initializers.zeros  # different from PyTorch?

  # msa_kernel_init = fixed_gaussian_init

  # TF/PyTorch: qkv is [D, 3*D], fan_in + fan_out = 4*D.
  # JAX: q, k, v each is [D, D], fan_in + fan_out = 2*D. So we compensate by scale=0.5
  qkv_kernel_init = functools.partial(nn.initializers.variance_scaling, 0.5, "fan_avg", "uniform")()
  out_kernel_init = nn.initializers.xavier_uniform()

  mlp_kernel_init = nn.initializers.xavier_uniform()
  mlp_bias_init = nn.initializers.zeros

else:
  raise NotImplementedError


class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  @nn.compact
  def __call__(self, x):
    return x


class Add2DPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.
  """
  sincos: bool
  use_cls_token: bool
  dtype: Any = jnp.float32

  def get_pos_emb(self, x):
    _, l, c = x.shape
    h = w = int(l**.5)
    assert h * w == l

    num_clstokens = 1 if self.use_cls_token else 0
    pos_emb_shape = (1, num_clstokens + h * w, c)  # (batch_size, seq_len, emb_dim).

    if not self.sincos:
      raise NotImplementedError
      init_fn = posemb_init
    else:
      pe_array = posembed_util.get_2d_sincos_pos_embed(c, (h, w), cls_token=self.use_cls_token)  # in numpy array
      init_fn = initializers_util.constant(value=pe_array, dtype=self.dtype)

    pe = t5x.layers.param_with_axes(
        'pos_embedding',
        init_fn,
        pos_emb_shape,
        jnp.float32,
        axes=('_null0', 'length', 'embed'))

    # kaiming: in MAE, we should always set posembed for cls_token as zero.
    # when loading for finetuning, this zero posembed can be tuned.
    # but this is not addressed here if sincos=False

    return pe

  @nn.compact
  def __call__(self, inputs):
    """Applies Add2DPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init.

    Args:
      inputs: Inputs to the layer.

    Returns:
      Output tensor with shape `(bs, timesteps, in_dim)`.
    """
    pe = self.get_pos_emb(inputs)
    pe = jax.lax.stop_gradient(pe) if self.sincos else self.pe

    if self.use_cls_token:
      output = inputs + pe[:, 1:, :]
    else:
      output = inputs + pe

    return output


class Add1DPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.
  """
  sincos: bool
  dtype: Any = jnp.float32
  posemb_init: Any = None

  def get_pos_emb(self, x):
    _, l, c = x.shape
    pos_emb_shape = (1, l, c)  # (batch_size, seq_len, emb_dim).

    if not self.sincos:
      init_fn = self.posemb_init
    else:
      raise NotImplementedError

    pe = t5x.layers.param_with_axes(
        'pos_embedding',
        init_fn,
        pos_emb_shape,
        jnp.float32,
        axes=('_null0', 'length', 'embed'))

    return pe

  @nn.compact
  def __call__(self, inputs):
    pe = self.get_pos_emb(inputs)    
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
  rescale_init: float = 1.

  @nn.compact
  def __call__(self, inputs, deterministic, mask=None):
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
      qkv_kernel_init=lambda *args: qkv_kernel_init(*args) * self.rescale_init,
      out_kernel_init=lambda *args: out_kernel_init(*args) * self.rescale_init,
    )
    # ----------------------------------------------------

    x = MsaBlock(
        dtype=self.dtype,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads,
    )(x, x, mask=mask)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    # droppath
    x = nn.Dropout(rate=self.droppath_rate, broadcast_dims=(1, 2), name='droppath_msa')(x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = t5x.layers.LayerNorm(dtype=self.dtype, axes=('embed',))(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate,
        kernel_init=lambda *args: mlp_kernel_init(*args) * self.rescale_init,
        bias_init=mlp_bias_init,
        )(y, deterministic=deterministic)
    # droppath
    y = nn.Dropout(rate=self.droppath_rate, broadcast_dims=(1, 2), name='droppath_mlp')(y, deterministic=deterministic)

    return x + y


class Encoder1DBlockCross(nn.Module):
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
  rescale_init: float = 1.

  @nn.compact
  def __call__(self, q, kv, *, deterministic):
    """Applies Encoder1DBlock module.

    Args:
      inputs: Inputs to the layer.
      deterministic: Dropout will not be applied when set to true.

    Returns:
      output after transformer encoder block.
    """
    assert self.dropout_rate == 0  # for now
    assert self.droppath_rate == 0  # for now

    # Attention block.
    assert q.ndim == 3, f'Expected (batch, seq, hidden) got {q.shape}'
    assert kv.ndim == 3, f'Expected (batch, seq, hidden) got {kv.shape}'
    kv = t5x.layers.LayerNorm(dtype=self.dtype, axes=('embed',))(kv)

    # ----------------------------------------------------
    # declare block type
    MsaBlock = functools.partial(
      t5x.layers.MultiHeadDotProductAttention,
      qkv_kernel_init=lambda *args: qkv_kernel_init(*args) * self.rescale_init,
      out_kernel_init=lambda *args: out_kernel_init(*args) * self.rescale_init,
    )
    # ----------------------------------------------------

    # first block: self-attention
    identity = q
    x = t5x.layers.LayerNorm(dtype=self.dtype, axes=('embed',))(q)
    x = MsaBlock(
        dtype=self.dtype,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads,
        name='SelfAtt'
    )(x, x)
    x = x + identity

    # second block: cross-attention
    identity = x
    x = t5x.layers.LayerNorm(dtype=self.dtype, axes=('embed',))(x)
    x = MsaBlock(
        dtype=self.dtype,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads,
        name='CrossAtt'
    )(x, kv)
    x = x + identity

    # MLP block.
    identity = x
    y = t5x.layers.LayerNorm(dtype=self.dtype, axes=('embed',))(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate,
        kernel_init=lambda *args: mlp_kernel_init(*args) * self.rescale_init,
        bias_init=mlp_bias_init,
        )(y, deterministic=deterministic)
    y = y + identity
    return y


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
  prefix: str = 'encoder'
  rescale_init: float = 1.0
  ln_pre: bool = False
  remat_policy: str = "none"

  @nn.compact
  def __call__(self, inputs, *, train, mask=None):
    """Applies Transformer model on the inputs.

    Args:
      inputs: Inputs to the layer.
      train: Set to `True` when training.

    Returns:
      output of a transformer encoder.
    """
    assert inputs.ndim == 3  # (batch, len, emb)

    BlockLayer = Encoder1DBlock
    if self.remat_policy not in (None, 'none'):
      if self.remat_policy == 'minimal':
        logging.info("activation checkpointing")
        policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
      else:
        policy = None
      BlockLayer = remat(  # pylint: disable=invalid-name
          Encoder1DBlock,
          prevent_cse=True,
          policy=policy,
          static_argnums=(1,)
      )  # "deterministic" is a static argument in Encoder1DBlock

    x = inputs

    if self.ln_pre:
      x = t5x.layers.LayerNorm(name=self.prefix + '_norm_pre', axes=('embed',))(x)

    for lyr in range(self.num_layers):
      deterministic=not train
      x = BlockLayer(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          droppath_rate=self.droppath_rate * lyr / (self.num_layers - 1) if self.droppath_rate > 0. else 0.,
          name=self.prefix + 'block_{:02d}'.format(lyr),
          num_heads=self.num_heads,
          layer_id=lyr,
          rescale_init=self.rescale_init,
        )(x, deterministic, mask)
    encoded = t5x.layers.LayerNorm(name=self.prefix + '_norm', axes=('embed',))(x)

    return encoded


class EncoderCross(nn.Module):
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
  prefix: str = 'encoder'
  rescale_init: float = 1.0

  @nn.compact
  def __call__(self, inputs, *, train):
    """Applies Transformer model on the inputs.

    Args:
      inputs: Inputs to the layer.
      train: Set to `True` when training.

    Returns:
      output of a transformer encoder.
    """
    assert isinstance(inputs, tuple)
    assert len(inputs) == 2

    q, kv = inputs
    assert q.ndim == 3  # (batch, len of q, emb)
    assert kv.ndim == 3  # (batch, len of kv, emb)

    x = q
    for lyr in range(self.num_layers):
      x = Encoder1DBlockCross(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          droppath_rate=self.droppath_rate * lyr / (self.num_layers - 1) if self.droppath_rate > 0. else 0.,
          name=self.prefix + 'block_{:02d}'.format(lyr),
          num_heads=self.num_heads,
          layer_id=lyr,
          rescale_init=self.rescale_init,
        )(q=x, kv=kv, deterministic=not train)
    encoded = t5x.layers.LayerNorm(name=self.prefix + '_norm', axes=('embed',))(x)

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


def random_mask(rng, x, mask_ratio, bias=None):
  """
  x: [N, L, C] input
  bias: [N, L], an additional map to the noise map (small is keep, large is remove)
  """

  N, L, _ = x.shape  # batch, length, dim
  len_keep = int(L * (1 - mask_ratio))

  noise = random.uniform(rng, shape=x.shape[:2])  

  if bias is not None:
    noise += bias

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


  # for debug
  # ndevice = jax.device_count()
  # print(ndevice)
  # B, L, C = x_masked.shape
  # nd = min(ndevice, B)
  # x_masked = jnp.reshape(x_masked, [nd, B // nd, L, C])
  # x_masked = jnp.concatenate([x_masked] * 2, axis=1)
  # x_masked = jnp.reshape(x_masked, [N * 2, L, C])
  # x_masked = jnp.reshape(x_masked, [x_masked.shape[0] // 2, 2, x_masked.shape[1], x_masked.shape[2]]).mean(axis=1)
  # x_masked = t5x.layers.with_sharding_constraint(x_masked, ('batch', 'length', 'embed'))
  

  # mask = jnp.concatenate([mask] * 2, axis=0)
  # ids_restore = jnp.concatenate([ids_restore] * 2, axis=0)


  return x_masked, mask, ids_restore


def random_mask_repeat(rng, x, mask_ratio, bias=None, mode="complementary", repeat=1):
  """
  x: [N, L, C] input
  bias: [N, L], an additional map to the noise map (small is keep, large is remove)
  """

  N, L, _ = x.shape  # batch, length, dim
  len_keep = int(L * (1 - mask_ratio))

  noise = random.uniform(rng, shape=x.shape[:2])  

  if bias is not None:
    noise += bias

  ids_shuffle = jnp.argsort(noise, axis=1)  # ascend: small is keep, large is remove
  ids_restore = jnp.argsort(ids_shuffle, axis=1)

  assert repeat * len_keep <= L

  all_mask = []
  all_x_masked = []
  all_ids_restore = []

  for i in range(repeat):
    # keep the first subset
    if mode == "complementary":
      ids_keep = ids_shuffle[:, i * len_keep:(i * len_keep + len_keep)]
    else:
      raise NotImplementedError
    x_masked = gather_by_einsum(x, ids_keep)

    x_masked = t5x.layers.with_sharding_constraint(x_masked, ('batch', 'length', 'embed'))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = jnp.ones([N, L])
    mask = t5x.layers.with_sharding_constraint(mask, ('batch', 'length'))
    if mode == "complementary":
      mask = mask.at[:, i * len_keep:(i * len_keep + len_keep)].set(0)
    else:
      raise NotImplementedError
    # unshuffle to get the binary mask
    mask = gather_by_einsum(mask, ids_restore)
    mask = t5x.layers.with_sharding_constraint(mask, ('batch', 'length'))

    all_mask.append(mask)
    all_x_masked.append(x_masked)
    all_ids_restore.append(ids_restore)
  
  def _concat_by_device(x_list):
    ## TODO support partition > 1
    shape = list(x_list[0].shape)
    nd = min(jax.device_count(), shape[0])
  
    x_list = [jnp.reshape(x, [nd, shape[0] // nd] + shape[1:]) for x in x_list]
    x = jnp.concatenate(x_list, axis=1)
    x = jnp.reshape(x, [shape[0] * len(x_list)] + shape[1:])
    
    return x

  all_x_masked = _concat_by_device(all_x_masked)
  all_mask = _concat_by_device(all_mask)
  all_ids_restore = _concat_by_device(all_ids_restore)

  # all_x_masked = jnp.concatenate(all_x_masked, axis=0)
  # all_mask = jnp.concatenate(all_mask, axis=0)
  # all_ids_restore = jnp.concatenate(all_ids_restore, axis=0)

  # all_x_masked = t5x.layers.with_sharding_constraint(all_x_masked, ('batch', 'length', 'embed'))
  # all_mask = t5x.layers.with_sharding_constraint(all_mask, ('batch', 'length'))
  # all_ids_restore = t5x.layers.with_sharding_constraint(all_ids_restore, ('batch', 'length'))

  return all_x_masked, all_mask, all_ids_restore


class LanguageTransformer(nn.Module):
  """LanguageTransformer."""

  mask_ratio: float
  sincos: bool
  vocab_size: int
  transformer: Any
  hidden_size: int
  dtype: Any = jnp.float32
  use_attention_mask: bool = False
  decoder: Any = None

  def setup(self):
    """
    declare all param layers based on inputs
    """
    # ------------------------
    # define encoder
    # ------------------------
    encoder_layers = {}
    encoder_layers['token_emb'] = t5x.layers.Embed(
      num_embeddings=self.vocab_size,
      features=self.hidden_size,
      embedding_init=fixed_gaussian_init,
      one_hot=True,
      axes=['classes', 'embed'],  # do not use 'vocab' 
      name='token_embedding')
    encoder_layers['pos_emb'] = Add1DPositionEmbs(sincos=self.sincos, posemb_init=fixed_gaussian_init, name='posembed_encoder')
    encoder_layers['blocks'] = Encoder(name='Transformer', **self.transformer, prefix='encoder')
    self.encoder_layers = encoder_layers

    # ------------------------
    # define decoder
    # ------------------------
    decoder_layers = {}
    decoder_layers['bottleneck'] = t5x.layers.Dense(
      features=self.decoder.hidden_size,
      kernel_init=mlp_kernel_init,
      bias_init=mlp_bias_init,
      kernel_axes=('mlp', 'embed'),  # 'mlp' is split first
      name='bottleneck')
    decoder_layers['mask_token'] = t5x.layers.param_with_axes(
      'mask_token', masktoken_init, (1, 1, self.decoder.hidden_size),
      jnp.float32, axes=('_null0', '_null1', 'embed'))
    decoder_layers['posemb'] = Add1DPositionEmbs(sincos=self.sincos, posemb_init=fixed_gaussian_init, name='posembed_decoder')
    if self.decoder.on_use:
      if self.decoder.cross_attention:
        decoder_layers['blocks'] = EncoderCross(name='TransformerDecoder', **self.decoder.transformer, prefix='decoder')
      else:
        decoder_layers['blocks'] = Encoder(name='TransformerDecoder', **self.decoder.transformer, prefix='decoder')
      decoder_layers['pred'] = t5x.layers.Dense(
        features=self.vocab_size,
        kernel_init=mlp_kernel_init,
        bias_init=mlp_bias_init,
        kernel_axes=('embed', 'classes'),  # 'mlp' is split first
        name='pred')
    self.decoder_layers = decoder_layers

  def compute_loss(self, txt, pred, mask, is_valid):
    """
    txt: [N, L]
    pred: [N, L, K]
    mask: [N, L], 0 is keep (known), 1 is remove (unknown)
    is_valid: [N, L], 1 is real, 0 is pad
    """    
    labels_one_hot = jax.nn.one_hot(txt, self.vocab_size)  # [N, L, K]
    loss = optax.softmax_cross_entropy(pred, labels_one_hot)  # [N, L]

    # has loss: mask=1 (unknown) and is_valid=1 (real)
    mask = jnp.logical_and(jnp.bool_(mask), jnp.bool_(is_valid))
    mask = jnp.float32(mask)

    loss = jnp.sum(loss * mask) / jnp.sum(mask)  # mean loss on removed patches

    return loss

  def apply_encoder(self, inputs, train):
    x = inputs
    
    x = self.encoder_layers['token_emb'](x)

    x = self.encoder_layers['pos_emb'](x)

    # masking: length -> length * mask_ratio
    # x, mask, ids_restore = random_mask(self.make_rng('dropout'), x, self.mask_ratio)
    mask, ids_restore = None, None

    # apply the encoder
    if self.use_attention_mask:
      _, L, _ = x.shape
      mask = jnp.tril(jnp.ones(shape=(L, L), dtype=jnp.float32)) # make a lower triangle
      mask = jnp.reshape(mask, (1, 1, L, L))
    else:
      mask = None

    x = self.encoder_layers['blocks'](x, train=train, mask=mask)

    return x, mask, ids_restore

  def apply_unshuffle(self, x, ids_restore):
    """bottleneck projection, unshuffle, add pos emb"""
    n, l = ids_restore.shape

    # apply the encoder-decoder bottleneck
    x = self.decoder_layers['bottleneck'](x)
    x_part = x  # take a copy

    # append mask token
    mask_token = self.decoder_layers['mask_token']
    mask_tokens = jnp.tile(mask_token, [n, ids_restore.shape[1] - x.shape[1], 1])
    x_ = jnp.concatenate([x, mask_tokens], axis=1)  # no cls token
    x_ = gather_by_einsum(x_, ids_restore)

    # add decoder posembed
    x_full = self.decoder_layers['posemb'](x_)

    return x_full, x_part

  def apply_decoder(self, x, train):

    # apply the decoder
    x = self.decoder_layers['blocks'](x, train=train)

    # apply the predictor
    x = self.decoder_layers['pred'](x)

    return x


class VisionTransformer(nn.Module):
  """VisionTransformer."""

  mask_ratio: float
  sincos: bool
  norm_pix_loss: bool
  patches: Any
  transformer: Any
  hidden_size: int
  classifier: str = 'token'
  dtype: Any = jnp.float32
  decoder: Any = None
  ln_pre: bool = False

  def patchify(self, imgs):
      """
      imgs: (N, H, W, 3)
      x: (N, L, patch_size**2 *3)
      """
      p, q = self.patches.size
      h, w = imgs.shape[1] // p, imgs.shape[2] // q 

      x = jnp.reshape(imgs, (imgs.shape[0], h, p, w, q, 3))
      x = jnp.einsum('nhpwqc->nhwpqc', x)
      x = jnp.reshape(x, (imgs.shape[0], h * w, p * q * 3))
      return x

  def unpatchify(self, x):
      """
      x: (N, L, patch_size**2 *3)
      imgs: (N, H, W, 3)
      """
      p, q = self.patches.size
      h = w = int(x.shape[1]**.5)

      x = jnp.reshape(x, (x.shape[0], h, w, p, q, 3))
      x = jnp.einsum('nhwpqc->nhpwqc', x)
      imgs = jnp.reshape(x, (x.shape[0], h * p, w * q, 3))
      return imgs

  def compute_loss(self, imgs, pred, mask):
    """
    imgs: [N, H, W, 3]
    pred: [N, L, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove, 
    """
    target = self.patchify(imgs)
    if self.norm_pix_loss:
      # target = jax.nn.normalize(target, axis=-1, epsilon=1.e-6)
      mean = jnp.mean(target, axis=-1, keepdims=True)
      var = jnp.var(target, axis=-1, keepdims=True)
      target = (target - mean) / (var + 1.e-6)**.5

    loss = jnp.square(pred - target)
    loss = jnp.mean(loss, axis=-1)  # [N, L], mean loss per patch

    loss = jnp.sum(loss * mask) / jnp.sum(mask)  # mean loss on removed patches
    return loss

  def visualization(self, imgs, pred, mask):
    """
    imgs: [N, H, W, 3]
    pred: [N, L, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove, 
    """
    imgs_pred = self.unpatchify(pred)

    mask = jnp.repeat(jnp.expand_dims(mask, axis=-1), repeats=pred.shape[-1], axis=-1)
    mask = self.unpatchify(mask)  # 0 is keep, 1 is remove
    imgs_mask = imgs * (1 - mask)

    imgs_plus = imgs * (1 - mask) + imgs_pred * mask

    imgs_vis = jnp.concatenate(
    [jnp.concatenate([imgs, imgs_mask], axis=2),
     jnp.concatenate([imgs_pred, imgs_plus], axis=2)],
    axis=1)
    return imgs_vis

  def apply_encoder(self, inputs, train, full_prob=0.0):
    use_cls_token = (self.classifier == 'token')
    assert use_cls_token  # kaiming: TODO: support both?

    x = self.encoder_layers['patch_emb'](inputs)

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    x = self.encoder_layers['pos_emb'](x)

    # masking: length -> length * mask_ratio
    mask_ratio = self.mask_ratio if train else 0.0
    assert full_prob == 0.0
    if full_prob > 0 and random.uniform(self.make_rng('dropout')) <= full_prob:
      mask_ratio = 0.0

    x, mask, ids_restore = random_mask(self.make_rng('dropout'), x, mask_ratio)
    n = x.shape[0]
    ids_restore = jnp.reshape(ids_restore, [n, h, w])  # carries the shape info

    if use_cls_token:
      cls = self.encoder_layers['cls_token']
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    # apply the encoder
    x = self.encoder_layers['blocks'](x, train=train)

    return x, mask, ids_restore

  def apply_encoder_multi(self, inputs, train, repeat_mode="none", repeat_sample=1, full_prob=0.0):
    use_cls_token = (self.classifier == 'token')
    assert use_cls_token  # kaiming: TODO: support both?

    x = self.encoder_layers['patch_emb'](inputs)

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    x = self.encoder_layers['pos_emb'](x)

    mask_ratio = self.mask_ratio if train else 0.0
    assert full_prob == 0.0
    if full_prob > 0 and random.uniform(self.make_rng('dropout')) <= full_prob: # not working for comp
      mask_ratio = 0.0
      repeat_sample = 1

    if train and repeat_mode != "none" and repeat_sample > 1:
      # masking: length -> length * mask_ratio
     
      x, mask, ids_restore = random_mask_repeat(
        self.make_rng('dropout'), x, mask_ratio, mode=repeat_mode, repeat=repeat_sample
      )
      ids_restore = jnp.reshape(ids_restore, [n * repeat_sample, h, w])  # carries the shape info
    else:
      # masking: length -> length * mask_ratio
      
      x, mask, ids_restore = random_mask(self.make_rng('dropout'), x, mask_ratio)
      ids_restore = jnp.reshape(ids_restore, [n, h, w])  # carries the shape info

    if use_cls_token:
      cls = self.encoder_layers['cls_token']
      cls = jnp.tile(cls, [n * repeat_sample, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    # apply the encoder
    x = self.encoder_layers['blocks'](x, train=train)

    return x, mask, ids_restore


  def apply_unshuffle(self, x, ids_restore):
    """bottleneck projection, unshuffle, add pos emb"""
    use_cls_token = (self.classifier == 'token')

    n, h, w = ids_restore.shape
    ids_restore = jnp.reshape(ids_restore, [n, h * w])

    # apply the encoder-decoder bottleneck
    x = self.decoder_layers['bottleneck'](x)
    x_part = x  # take a copy

    # append mask token
    num_clstokens = 1 if use_cls_token else 0
    mask_token = self.decoder_layers['mask_token']
    mask_tokens = jnp.tile(mask_token, [n, ids_restore.shape[1] + num_clstokens - x.shape[1], 1])
    x_ = jnp.concatenate([x[:, num_clstokens:, :], mask_tokens], axis=1)  # no cls token
    x_ = gather_by_einsum(x_, ids_restore)

    # add decoder posembed (before cls token)
    x_ = self.decoder_layers['pos_emb'](x_)

    x_full = jnp.concatenate([x[:, :num_clstokens, :], x_], axis=1)  # append cls token
    return x_full, x_part

  def apply_decoder(self, x, train):
    use_cls_token = (self.classifier == 'token')
    num_clstokens = 1 if use_cls_token else 0

    # apply the decoder
    x = self.decoder_layers['blocks'](x, train=train)

    # apply the predictor
    x = self.decoder_layers['pred'](x)

    # remove cls token
    pred = x[:, num_clstokens:, :]

    return pred

  def setup(self):
    """
    declare all param layers based on inputs
    """
    # ------------------------
    # define encoder
    # ------------------------
    use_cls_token = (self.classifier == 'token')
    assert use_cls_token  # kaiming: TODO: support both?

    encoder_layers = {}  # cannot directly declare self.encoder_layers
    encoder_layers['patch_emb'] = t5x.layers.Conv(
        features=self.hidden_size,
        kernel_size=self.patches.size,
        strides=self.patches.size,
        padding='VALID',
        name='embedding',
        kernel_init=patch_kernel_init,
        bias_init=patch_bias_init,
        kernel_axes=('_null0', '_null1', '_null2', 'embed'))
    encoder_layers['pos_emb'] = Add2DPositionEmbs(sincos=self.sincos, use_cls_token=use_cls_token, name='posembed_encoder')
    if use_cls_token:
      encoder_layers['cls_token'] = t5x.layers.param_with_axes('cls', clstoken_init, (1, 1, self.hidden_size), jnp.float32, axes=('_null0', '_null1', 'embed'))
    encoder_layers['blocks'] = Encoder(name='Transformer', **self.transformer, prefix='encoder', ln_pre=self.ln_pre)
    self.encoder_layers = encoder_layers

    # ------------------------
    # define decoder
    # ------------------------
    decoder_layers = {}
    decoder_layers['bottleneck'] = t5x.layers.Dense(
      features=self.decoder.hidden_size,
      kernel_init=mlp_kernel_init,
      bias_init=mlp_bias_init,
      kernel_axes=('mlp', 'embed'),  # 'mlp' is split first
      name='bottleneck')
    decoder_layers['mask_token'] = t5x.layers.param_with_axes(
      'mask_token', masktoken_init, (1, 1, self.decoder.hidden_size),
      jnp.float32, axes=('_null0', '_null1', 'embed'))
    decoder_layers['pos_emb'] = Add2DPositionEmbs(sincos=self.sincos, use_cls_token=use_cls_token, name='posembed_decoder')
    if self.decoder.on_use:
      if self.decoder.cross_attention:
        decoder_layers['blocks'] = EncoderCross(name='TransformerDecoder', **self.decoder.transformer, prefix='decoder')
      else:
        decoder_layers['blocks'] = Encoder(name='TransformerDecoder', **self.decoder.transformer, prefix='decoder')
      decoder_layers['pred'] = t5x.layers.Dense(
        features=self.patches.size[0] * self.patches.size[1] * 3,
        kernel_init=mlp_kernel_init,
        bias_init=mlp_bias_init,
        kernel_axes=('embed', 'classes'),  # 'mlp' is split first
        name='pred')
    self.decoder_layers = decoder_layers

  
class ImageTextLearner(nn.Module):
  """ContrastiveLearner with Vision Transformer
  """
  config: Any = None  # model config
  dtype: Any = jnp.float32

  def get_config_img(self):
    cfg = self.config.model_img.copy_and_resolve_references()  # copy
    cfg.name = 'img_encoder'  # force name
    return cfg

  def get_config_txt(self):
    cfg = self.config.model_txt.copy_and_resolve_references()  # copy
    cfg.name = 'txt_encoder'  # force name
    return cfg

  def setup(self):
    self.img_encoder = VisionTransformer(**self.get_config_img())
    self.txt_encoder = LanguageTransformer(**self.get_config_txt())

    if self.config.clr.get("momentum_queue", 0) > 0:
      self.img_queue = MemQueue(queue_size=self.config.clr.momentum_queue, queue_dim=self.config.clr.proj_dim_out)
      self.txt_queue = MemQueue(queue_size=self.config.clr.momentum_queue, queue_dim=self.config.clr.proj_dim_out)

  def apply_projection_head(self, z, prefix):
    clr = self.config.clr
    for i in range(clr.proj_layers - 1):
      z = t5x.layers.Dense(
        features=clr.proj_dim_hidden,
        dtype=self.dtype,
        kernel_init=mlp_kernel_init,
        bias_init=mlp_bias_init,
        kernel_axes=('_null0', '_null1'),
        name='{}_mlp{}'.format(prefix, i))(z)
      z = nn.gelu(z)
    z = t5x.layers.Dense(
      features=clr.proj_dim_out,
      dtype=self.dtype,
      kernel_init=mlp_kernel_init,
      bias_init=mlp_bias_init,
      use_bias=clr.proj_out_bias,
      kernel_axes=('_null0', '_null1'),
      name='{}_mlp{}'.format(prefix, clr.proj_layers))(z)
    return z
  
  def compute_contrastive_loss(self, z0, z1):
    clr = self.config.clr

    if clr.tau_learnable:
      logit_scale = t5x.layers.param_with_axes(
          'logit_scale', initializers_util.constant(value=math.log(1 / 0.07)),
          (1,), jnp.float32, axes=('_null0',))
      logit_scale = jnp.clip(logit_scale, 0, math.log(100))
      tau = 1 / jnp.exp(logit_scale)
    else:
      tau = clr.tau
      logit_scale = None

    def _get_logits(a, b, logit_scale):
      x = jnp.einsum('nc,mc->nm', a, b)
      
      if clr.tau_learnable:
        x *= jnp.exp(logit_scale)
      else:
        x /= clr.tau
      
      return x
    
    logits = _get_logits(z0, z1, logit_scale)
    logging.info('logits.shape: {}'.format(logits.shape))
    n, m = logits.shape[0], logits.shape[1]
    # assert n == m * repeat
    repeat = n // m 

    if clr.get("momentum_queue", 0) > 0:
      assert repeat == 1
      q1 = self.txt_queue.get_queue()
      q1 = self.txt_queue.get_queue()
      logits_z0z1 = _get_logits(z0, z1, logit_scale) 
      logits_z0q1 = _get_logits(z0, q1, logit_scale)
      labels_one_hot = jnp.eye(n, n + q1.shape[0])
      loss01 = clr_loss(
        config=clr,
        logits=jnp.concatenate([logits_z0z1, logits_z0q1], axis=1),
        labels=labels_one_hot,
      )
      

      q0 = self.img_queue.get_queue()
      q0 = jax.lax.stop_gradient(q0) 
      logits_z1z0 = _get_logits(z1, z0, logit_scale) 
      logits_z1q0 = _get_logits(z1, q0, logit_scale)
      labels_one_hot = jnp.eye(n, n + q0.shape[0])
      loss10 = clr_loss(
        config=clr,
        logits=jnp.concatenate([logits_z1z0, logits_z1q0], axis=1),
        labels=labels_one_hot,
      )

      self.img_queue.update_queue(z0)
      self.txt_queue.update_queue(z1)
    else:
      labels_one_hot = jnp.eye(m)

      if repeat == 1:
        loss01 = clr_loss(config=clr, logits=logits, labels=labels_one_hot)
        loss10 = clr_loss(config=clr, logits=logits.transpose(), labels=labels_one_hot)
      else:
        #from jax.experimental.host_callback import call

        nd = min(jax.device_count(), m)
        labels_one_hot = jnp.reshape(labels_one_hot, [nd, 1, m // nd, m])
        labels_one_hot = jnp.tile(labels_one_hot, [1, repeat, 1, 1])
        logits = jnp.reshape(logits, [nd, repeat, m // nd, m])
        loss01 = clr_loss(config=clr, logits=logits, labels=labels_one_hot)

        if clr.get("repeat_avg", False):
          # (nd, repeat, m // nd, m) --> (nd, m // nd, m)
          logits = jnp.mean(logits, axis=1)
          logits = jnp.reshape(logits, [m, m])
          labels_one_hot = jnp.eye(m)
          loss10 = clr_loss(
            config=clr,
            logits=logits.transpose(),
            labels=labels_one_hot,
          )
        else:
          # -> (nd, m/nd, repeat, m)
          logits = jnp.transpose(logits, axes=[0, 2, 1, 3])
          labels_one_hot = jnp.transpose(labels_one_hot, axes=[0, 2, 1, 3])
          logits = jnp.reshape(logits, [m, repeat, m])
          labels_one_hot = jnp.reshape(labels_one_hot, [m, repeat, m])
          # call(lambda x: print(x.shape, x), labels_one_hot)
          loss10 = clr_loss(
            config=clr,
            logits=logits.transpose(),
            labels=labels_one_hot.transpose(),
          )

        #loss10 = 0.0

    loss = (loss01 + loss10) / 2
    return loss, tau, loss01, loss10

  @nn.compact
  def __call__(self, inputs, *, train, encode_img=True, encode_txt=True):
    if encode_img:
      img = inputs['image']
    
    if encode_txt:
      txt = inputs['txt']
      is_valid = inputs['txt_is_valid']

    # apply both encoders
    if encode_img:
      repeat_mode = self.config.get("mask_repeat_mode", "none")
      repeat = self.config.get("mask_repeat", 1)
      full_prob = self.config.get("mask_full_prob", 0.0)
      if train and repeat_mode != "none":
        # print(img.shape)
        x_img, mask_img, ids_restore_img = self.img_encoder.apply_encoder_multi(
          img, train=train, repeat_mode=repeat_mode, repeat_sample=repeat, full_prob=full_prob,
        )
        # print(x_img.shape, mask_img.shape, ids_restore_img.shape)
      else:
        x_img, mask_img, ids_restore_img = self.img_encoder.apply_encoder(img, train=train, full_prob=full_prob)
    if encode_txt:
      x_txt, mask_txt, ids_restore_txt = self.txt_encoder.apply_encoder(txt, train=train)

    # apply contrastive learning (clip-like)
    if self.config.clr.clr_loss:
      if encode_img:
        if self.config.clr.get("img_cls_token", False):
          z_img = x_img[:, 0, :] # use cls_token
        else:
          z_img = x_img.mean(axis=1)  # avearge pool anyway
        z_img = self.apply_projection_head(z_img, prefix='img')
        z_img /= jnp.linalg.norm(z_img, axis=-1, keepdims=True) + 1e-8
      if encode_txt:
        if self.txt_encoder.use_attention_mask:
          ids_eos = jnp.argmax(jnp.cumsum(is_valid - 1e-6, axis=-1), axis=-1)  # find the last one
          ids_eos = ids_eos[:, None]
          z_txt = gather_by_einsum(x_txt, ids_eos).squeeze(axis=1)
        else:
          z_txt = x_txt[:, 0, :]  # cls token anyway
        z_txt = self.apply_projection_head(z_txt, prefix='txt')
        z_txt /= jnp.linalg.norm(z_txt, axis=-1, keepdims=True) + 1e-8
      if encode_img and encode_txt:
        # z_img = z_img.reshape([z_img.shape[0] // 2, 2, z_img.shape[1]]).mean(axis=1)
        loss_clr, tau, loss01, loss10 = self.compute_contrastive_loss(z_img, z_txt)
      else:
        loss_clr = 0
        tau = 0
        loss01 = 0
        loss10 = 0
    else:
      raise NotImplementedError
      loss_clr = 0

    # apply both decoders
    
    # x_txt_full, x_txt_part = self.txt_encoder.apply_unshuffle(x_txt, ids_restore_txt)

    if encode_img and self.img_encoder.decoder.on_use:
      # raise NotImplementedError
      x_img_full, x_img_part = self.img_encoder.apply_unshuffle(x_img, ids_restore_img)
      pred_img = self.img_encoder.apply_decoder((x_img_full, x_txt_part) if self.img_encoder.decoder.cross_attention else x_img_full, train=train)
    else:
      pred_img = None

    if self.txt_encoder.decoder.on_use:
      raise NotImplementedError
      # pred_txt = self.txt_encoder.apply_decoder((x_txt_full, x_img_part) if self.txt_encoder.decoder.cross_attention else x_txt_full, train=train)
    else:
      pred_txt = None

    # compute losses
    if encode_img and self.img_encoder.decoder.on_use:
      #raise NotImplementedError
      if train and repeat > 1:
        img = jnp.tile(img, [repeat, 1, 1])
      loss_img = self.img_encoder.compute_loss(img, pred_img, mask_img)
      vis = self.img_encoder.visualization(img, pred_img, mask_img)
    else:
      loss_img = 0
      vis = None

    if self.txt_encoder.decoder.on_use:
      raise NotImplementedError
      # loss_txt = self.txt_encoder.compute_loss(txt, pred_txt, mask_txt, is_valid)
    else:
      loss_txt = 0

    loss_tot = loss_img * self.img_encoder.decoder.loss_weight + loss_txt * self.txt_encoder.decoder.loss_weight + loss_clr

    artifacts = {
      'loss': loss_img,  # always plot loss_img in the 'loss' metric
      'loss_clr': loss_clr,
      'loss_img': loss_img,
      'loss_txt': loss_txt,
      'loss_tot': loss_tot,
      'loss_clr01': loss01,
      'loss_clr10': loss10,
      'tau': tau}
    
    if not train and encode_img:
      artifacts['z_img'] = z_img
    if not train and encode_txt:
      artifacts['z_txt'] = z_txt

    return loss_tot, vis, artifacts


def clr_loss(config, logits, labels):
  loss_type = config.get("loss_type", "ce")
  if loss_type == "ce":
    return optax.softmax_cross_entropy(logits=logits, labels=labels).mean()
  elif loss_type == "focal":
    gamma = config.get("focal_gamma", 2.0)
    alpha = config.get("focal_alpha", 0.25)
    norm = config.get("focal_norm", "sigmoid")
    logging.info(f"use focal loss with gamma{gamma} and alpah{alpha} and norm{norm}")

    if norm == "sigmoid":
      p = jax.nn.sigmoid(logits)
      ce_loss = optax.sigmoid_binary_cross_entropy(logits, labels)
    elif norm == "softmax":
      p = jax.nn.softmax(logits)
      ce_loss = -labels * jax.nn.log_softmax(logits, axis=-1)
    else:
      raise NotImplementedError
    p_t = p * labels + (1 - p) * (1 - labels)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
      alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
      loss = alpha_t * loss
    return jnp.sum(ce_loss, axis=1).mean()
  else:
    raise NotImplementedError

