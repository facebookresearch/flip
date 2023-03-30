# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, Callable, Optional, Tuple

from absl import logging
import math

import jax
import jax.numpy as jnp
import jax.random as random

import flax.linen as nn
from flax.linen.partitioning import remat

import t5x.layers

from utils import posembed_util
from utils import initializers_util


Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

# init hacks
INIT_VER = "mae_jax_v2"

fixed_gaussian_init = nn.initializers.normal(stddev=0.02)
if INIT_VER == "mae_jax_v2":
    clstoken_init = fixed_gaussian_init
    masktoken_init = fixed_gaussian_init
    posemb_init = fixed_gaussian_init  # not used if sincos

    patch_kernel_init = initializers_util.patch_kernel()
    patch_bias_init = nn.initializers.zeros  # different from PyTorch?

    # TF/PyTorch: qkv is [D, 3*D], fan_in + fan_out = 4*D.
    # JAX: q, k, v each is [D, D], fan_in + fan_out = 2*D. So we compensate by scale=0.5
    qkv_kernel_init = functools.partial(
        nn.initializers.variance_scaling, 0.5, "fan_avg", "uniform"
    )()
    out_kernel_init = nn.initializers.xavier_uniform()

    mlp_kernel_init = nn.initializers.xavier_uniform()
    mlp_bias_init = nn.initializers.zeros

else:
    raise NotImplementedError


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    kernel_init: Callable[
        [PRNGKey, Shape, Dtype], Array
    ] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(
        stddev=1e-6
    )

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        """Applies Transformer MlpBlock module."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = t5x.layers.Dense(
            features=self.mlp_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            kernel_axes=("embed", "mlp"),
            name="Dense_0",
        )(inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = t5x.layers.with_sharding_constraint(x, ("batch", "length", "mlp"))
        output = t5x.layers.Dense(
            features=actual_out_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            kernel_axes=("mlp", "embed"),
            name="Dense_1",
        )(x)
        output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=deterministic)
        return output


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
      mlp_dim: dimension of the mlp on top of attention block.
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
      dtype: the dtype of the computation (default: float32).
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout for attention heads.
      layer_id: layer id.
    """

    mlp_dim: int
    num_heads: int
    dtype: Dtype = jnp.float32
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    droppath_rate: float = 0.0
    layer_id: int = None
    rescale_init: float = 1.0

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
        assert inputs.ndim == 3, f"Expected (batch, seq, hidden) got {inputs.shape}"
        x = t5x.layers.LayerNorm(dtype=self.dtype, axes=("embed",))(inputs)

        # ----------------------------------------------------
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
        x = nn.Dropout(
            rate=self.droppath_rate, broadcast_dims=(1, 2), name="droppath_msa"
        )(x, deterministic=deterministic)
        x = x + inputs

        # MLP block.
        y = t5x.layers.LayerNorm(dtype=self.dtype, axes=("embed",))(x)
        y = MlpBlock(
            mlp_dim=self.mlp_dim,
            dtype=self.dtype,
            dropout_rate=self.dropout_rate,
            kernel_init=lambda *args: mlp_kernel_init(*args) * self.rescale_init,
            bias_init=mlp_bias_init,
        )(y, deterministic=deterministic)
        # droppath
        y = nn.Dropout(
            rate=self.droppath_rate, broadcast_dims=(1, 2), name="droppath_mlp"
        )(y, deterministic=deterministic)

        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation.

    Attributes:
      num_layers: number of layers
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout rate in self attention.
      droppath_rate: drop path rate.
      prefix: prefix of block name.
      remat_policy: remat policy, e.g. activation checkpointing.
    """

    num_layers: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    droppath_rate: float = 0.0
    prefix: str = "encoder"
    rescale_init: float = 1.0
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
        if self.remat_policy not in (None, "none"):
            logging.info(f"remat policy: {self.remat_policy}")
            if self.remat_policy == "minimal":
                policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
            else:
                policy = None
            logging.info(f"activation checkpointing {self.remat_policy}")
            BlockLayer = remat(  # pylint: disable=invalid-name
                Encoder1DBlock, prevent_cse=True, policy=policy, static_argnums=(1,)
            )  # "deterministic" is a static argument in Encoder1DBlock

        x = inputs

        for lyr in range(self.num_layers):
            deterministic = not train
            x = BlockLayer(
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                droppath_rate=self.droppath_rate * lyr / (self.num_layers - 1)
                if self.droppath_rate > 0.0
                else 0.0,
                name=self.prefix + "block_{:02d}".format(lyr),
                num_heads=self.num_heads,
                layer_id=lyr,
                rescale_init=self.rescale_init,
            )(x, deterministic, mask)
        encoded = t5x.layers.LayerNorm(name=self.prefix + "_norm", axes=("embed",))(x)

        return encoded


def gather_by_einsum(x, ids):
    """kaiming: vmap + gather is slow with pjit; use einsum instead
    Args:
      x: [N, L, ...]
      ids: [N, K]
    """
    mat = jax.nn.one_hot(ids, x.shape[1])  # [N, K, L]
    x = jnp.einsum("nl...,nkl->nk...", x, mat)
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

    x_masked = t5x.layers.with_sharding_constraint(
        x_masked, ("batch", "length", "embed")
    )

    # generate the binary mask: 0 is keep, 1 is remove
    mask = jnp.ones([N, L])
    mask = t5x.layers.with_sharding_constraint(mask, ("batch", "length"))
    mask = mask.at[:, :len_keep].set(0)
    # unshuffle to get the binary mask
    mask = gather_by_einsum(mask, ids_restore)
    mask = t5x.layers.with_sharding_constraint(mask, ("batch", "length"))

    return x_masked, mask, ids_restore


class LanguageTransformer(nn.Module):
    """Language Transformer."""

    mask_ratio: float
    sincos: bool
    vocab_size: int
    transformer: Any
    hidden_size: int
    dtype: Any = jnp.float32

    def setup(self):
        """
        declare all param layers based on inputs
        """
        # ------------------------
        # define encoder
        # ------------------------
        encoder_layers = {}
        encoder_layers["token_emb"] = t5x.layers.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size,
            embedding_init=fixed_gaussian_init,
            one_hot=True,
            axes=["classes", "embed"],  # do not use 'vocab'
            name="token_embedding",
        )
        encoder_layers["pos_emb"] = posembed_util.Add1DPositionEmbs(
            sincos=self.sincos, posemb_init=fixed_gaussian_init, name="posembed_encoder"
        )
        encoder_layers["blocks"] = Encoder(
            name="Transformer", **self.transformer, prefix="encoder"
        )
        self.encoder_layers = encoder_layers

    def apply_encoder(self, inputs, train, is_valid=None):
        x = inputs

        x = self.encoder_layers["token_emb"](x)
        x = self.encoder_layers["pos_emb"](x)

        mask_ratio = self.mask_ratio if train else 0.0
        if mask_ratio > 0:
            raise NotImplementedError
        else:
            mask, ids_restore = None, None

        x = self.encoder_layers["blocks"](x, train=train)

        return x, mask, ids_restore


class VisionTransformer(nn.Module):
    """Vision Transformer."""

    mask_ratio: float
    sincos: bool
    patches: Any
    transformer: Any
    hidden_size: int
    dtype: Any = jnp.float32
    use_cls_token: bool = True

    def patchify(self, imgs):
        """
        imgs: (N, H, W, 3)
        x: (N, L, patch_size**2 *3)
        """
        p, q = self.patches.size
        h, w = imgs.shape[1] // p, imgs.shape[2] // q

        x = jnp.reshape(imgs, (imgs.shape[0], h, p, w, q, 3))
        x = jnp.einsum("nhpwqc->nhwpqc", x)
        x = jnp.reshape(x, (imgs.shape[0], h * w, p * q * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, H, W, 3)
        """
        p, q = self.patches.size
        h = w = int(x.shape[1] ** 0.5)

        x = jnp.reshape(x, (x.shape[0], h, w, p, q, 3))
        x = jnp.einsum("nhwpqc->nhpwqc", x)
        imgs = jnp.reshape(x, (x.shape[0], h * p, w * q, 3))
        return imgs

    def apply_encoder(self, inputs, train):
        x = self.encoder_layers["patch_emb"](inputs)

        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        x = self.encoder_layers["pos_emb"](x)

        # masking: length -> length * mask_ratio
        mask_ratio = self.mask_ratio if train else 0.0

        x, mask, ids_restore = random_mask(self.make_rng("dropout"), x, mask_ratio)
        n = x.shape[0]
        ids_restore = jnp.reshape(ids_restore, [n, h, w])  # carries the shape info

        if self.use_cls_token:
            cls = self.encoder_layers["cls_token"]
            cls = jnp.tile(cls, [n, 1, 1])
            x = jnp.concatenate([cls, x], axis=1)

        # apply the encoder
        x = self.encoder_layers["blocks"](x, train=train)

        return x, mask, ids_restore

    def setup(self):
        """
        declare all param layers based on inputs
        """
        # ------------------------
        # define encoder
        # ------------------------
        assert self.use_cls_token

        encoder_layers = {}  # cannot directly declare self.encoder_layers
        encoder_layers["patch_emb"] = t5x.layers.Conv(
            features=self.hidden_size,
            kernel_size=self.patches.size,
            strides=self.patches.size,
            padding="VALID",
            name="embedding",
            kernel_init=patch_kernel_init,
            bias_init=patch_bias_init,
            kernel_axes=("_null0", "_null1", "_null2", "embed"),
        )
        encoder_layers["pos_emb"] = posembed_util.Add2DPositionEmbs(
            sincos=self.sincos,
            use_cls_token=self.use_cls_token,
            name="posembed_encoder",
        )
        if self.use_cls_token:
            encoder_layers["cls_token"] = t5x.layers.param_with_axes(
                "cls",
                clstoken_init,
                (1, 1, self.hidden_size),
                jnp.float32,
                axes=("_null0", "_null1", "embed"),
            )
        encoder_layers["blocks"] = Encoder(
            name="Transformer",
            **self.transformer,
            prefix="encoder",
        )
        self.encoder_layers = encoder_layers


class FLIP(nn.Module):
    """
    Model builder for Fast Language-Image Pre-training (FLIP).

    "Scaling Language-Image Pre-training via Masking"
    Yanghao Li*, Haoqi Fan*, Ronghang Hu*, Christoph Feichtenhofer†, Kaiming He†
    https://arxiv.org/abs/2212.00794
    """

    config: Any = None
    dtype: Any = jnp.float32

    def get_config_img(self):
        cfg = self.config.model_img.copy_and_resolve_references()  # copy
        return cfg

    def get_config_txt(self):
        cfg = self.config.model_txt.copy_and_resolve_references()  # copy
        return cfg

    def setup(self):
        self.img_encoder = VisionTransformer(**self.get_config_img())
        self.txt_encoder = LanguageTransformer(**self.get_config_txt())

    def apply_projection_head(self, z, prefix):
        clr = self.config.clr
        z = t5x.layers.Dense(
            features=clr.proj_dim_out,
            dtype=self.dtype,
            kernel_init=mlp_kernel_init,
            bias_init=mlp_bias_init,
            use_bias=clr.proj_out_bias,
            kernel_axes=("_null0", "_null1"),
            name="{}_mlp1".format(prefix),
        )(z)
        return z

    def compute_contrastive_loss(self, z0, z1):
        clr = self.config.clr

        if clr.tau_learnable:
            logit_scale = t5x.layers.param_with_axes(
                "logit_scale",
                initializers_util.constant(value=math.log(1 / 0.07)),
                (1,),
                jnp.float32,
                axes=("_null0",),
            )
            logit_scale = jnp.clip(logit_scale, 0, math.log(100))
            scale = jnp.exp(logit_scale)
            tau = 1 / scale
        else:
            tau = clr.tau
            scale = 1 / tau
            logit_scale = None

        # memory-efficient implementation
        logits = jnp.einsum("nc,mc->nm", z0, z1)
        logging.info("logits.shape: {}".format(logits.shape))
        logits *= scale

        # ---------------------------------------------------------------------------
        logits_pos = jnp.einsum(
            "nc,nc->n", z0, z1
        )  # easier to take the diagonal (positive)
        logits_pos *= scale

        # hand-written log_softmax
        # we do not need to shift x_max as it is well-bound after l2-normalization
        exp_logits = jnp.exp(logits)
        logsumexp_logits01 = jnp.log(jnp.sum(exp_logits, axis=-1))  # [N,]
        logsumexp_logits10 = jnp.log(jnp.sum(exp_logits, axis=0))  # [N,]

        loss01 = -(logits_pos - logsumexp_logits01)  # [N,]
        loss10 = -(logits_pos - logsumexp_logits10)  # [N,]

        loss01 = loss01.mean()
        loss10 = loss10.mean()

        loss = (loss01 + loss10) / 2
        return loss, tau

    @nn.compact
    def __call__(self, inputs, *, train, encode_img=True, encode_txt=True):
        if encode_img:
            img = inputs["image"]

        if encode_txt:
            txt = inputs["txt"]
            is_valid = inputs["txt_is_valid"]

        # apply both encoders
        if encode_img:
            x_img, _, _ = self.img_encoder.apply_encoder(img, train=train)
        if encode_txt:
            x_txt, _, _ = self.txt_encoder.apply_encoder(
                txt, train=train, is_valid=is_valid
            )

        # apply contrastive learning
        if self.config.clr.clr_loss:
            if encode_img:
                if not self.config.clr.img_avg_token:
                    z_img = x_img[:, 0, :]  # use cls_token
                else:
                    z_img = x_img.mean(axis=1)
                z_img = self.apply_projection_head(z_img, prefix="img")

                z_img /= jnp.linalg.norm(z_img, axis=-1, keepdims=True) + 1e-8
            if encode_txt:
                if not self.config.clr.txt_avg_token:
                    z_txt = x_txt[:, 0, :]
                else:
                    z_txt = x_txt.mean(axis=1)

                z_txt = self.apply_projection_head(z_txt, prefix="txt")
                z_txt /= jnp.linalg.norm(z_txt, axis=-1, keepdims=True) + 1e-8

            if encode_img and encode_txt:
                loss_clr, tau = self.compute_contrastive_loss(z_img, z_txt)
            else:
                loss_clr = 0
                tau = 0
        else:
            raise NotImplementedError

        artifacts = {"loss": loss_clr, "tau": tau}

        if not train and encode_img:
            artifacts["z_img"] = z_img
        if not train and encode_txt:
            artifacts["z_txt"] = z_txt

        return loss_clr, artifacts
