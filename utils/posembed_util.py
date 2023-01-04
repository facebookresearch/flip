# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from typing import Any, Tuple
import numpy as np
import flax.linen as nn
import jax.numpy as jnp

import jax
import t5x.layers

from utils import initializers_util

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


class Add2DPositionEmbs(nn.Module):
    """Adds 2D positional embeddings to the inputs."""

    sincos: bool
    use_cls_token: bool
    dtype: Any = jnp.float32

    def get_pos_emb(self, x):
        _, l, c = x.shape
        h = w = int(l**0.5)
        assert h * w == l

        num_clstokens = 1 if self.use_cls_token else 0
        # (batch_size, seq_len, emb_dim).
        pos_emb_shape = (1, num_clstokens + h * w, c)

        if not self.sincos:
            raise NotImplementedError
        else:
            pe_array = get_2d_sincos_pos_embed(c, (h, w), cls_token=self.use_cls_token)
            init_fn = initializers_util.constant(value=pe_array, dtype=self.dtype)

        pe = t5x.layers.param_with_axes(
            "pos_embedding",
            init_fn,
            pos_emb_shape,
            jnp.float32,
            axes=("_null0", "length", "embed"),
        )

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
    """Adds (optionally learned) positional embeddings to the inputs."""

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
            "pos_embedding",
            init_fn,
            pos_emb_shape,
            jnp.float32,
            axes=("_null0", "length", "embed"),
        )

        return pe

    @nn.compact
    def __call__(self, inputs):
        pe = self.get_pos_emb(inputs)
        output = inputs + pe
        return output


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int tuple of the grid, (height, width)
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    h, w = grid_size

    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_w, emb_h], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
