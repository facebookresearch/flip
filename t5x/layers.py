# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# --------------------------------------------------------
# References:
# The code is adapted and modified from https://github.com/google-research/t5x/tree/main/t5x
# LICENSE: https://github.com/google-research/t5x/blob/2a62e14fd2806a28c8b24c7674fdd5423aa95e3d/LICENSE
# --------------------------------------------------------


"""Dense attention classes and mask/weighting functions."""

# pylint: disable=attribute-defined-outside-init,g-bare-generic

import dataclasses
import functools
import operator
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as np


# from flax.linen.partitioning import param_with_axes, with_sharding_constraint
param_with_axes = nn_partitioning.param_with_axes
variable_with_axes = nn_partitioning.variable_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint


# Type annotations
Array = jnp.ndarray
Axes = Union[int, Iterable[int]]
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
Activation = Callable[..., Array]
PrecisionLike = Union[
    None, str, lax.Precision, Tuple[str, str], Tuple[lax.Precision, lax.Precision]
]
# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]

default_embed_init = nn.initializers.variance_scaling(
    1.0, "fan_in", "normal", out_axis=0
)


def dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array] = None,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: DType = jnp.float32,
    float32_logits: bool = False,
):
    """Computes dot-product attention given query, key, and value.

    This is the core function for applying attention based on
    https://arxiv.org/abs/1706.03762. It calculates the attention weights given
    query and key and combines the values using the attention weights.

    Args:
      query: queries for calculating attention with shape of `[batch, q_length,
        num_heads, qk_depth_per_head]`.
      key: keys for calculating attention with shape of `[batch, kv_length,
        num_heads, qk_depth_per_head]`.
      value: values to be used in attention with shape of `[batch, kv_length,
        num_heads, v_depth_per_head]`.
      bias: bias for the attention weights. This should be broadcastable to the
        shape `[batch, num_heads, q_length, kv_length]` This can be used for
        incorporating causal masks, padding masks, proximity bias, etc.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      dtype: the dtype of the computation (default: float32)
      float32_logits: bool, if True then compute logits in float32 to avoid
        numerical issues with bfloat16.

    Returns:
      Output of shape `[batch, length, num_heads, v_depth_per_head]`.
    """
    assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
    assert (
        query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
    ), "q, k, v batch dims must match."
    assert (
        query.shape[-2] == key.shape[-2] == value.shape[-2]
    ), "q, k, v num_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."
    assert query.shape[-1] == key.shape[-1], "q, k depths must match."

    # Casting logits and softmax computation for float32 for model stability.
    if float32_logits:
        query = query.astype(jnp.float32)
        key = key.astype(jnp.float32)

    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)
    # `attn_weights`: [batch, num_heads, q_length, kv_length]
    attn_weights = jnp.einsum("bqhd,bkhd->bhqk", query, key)

    # Apply attention bias: masking, dropout, proximity bias, etc.
    if bias is not None:
        attn_weights = attn_weights + bias.astype(attn_weights.dtype)

    # Normalize the attention weights across `kv_length` dimension.
    attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

    # Apply attention dropout.
    if not deterministic and dropout_rate > 0.0:
        raise NotImplementedError
        keep_prob = 1.0 - dropout_rate
        # T5 broadcasts along the "length" dim, but unclear which one that
        # corresponds to in positional dimensions here, assuming query dim.
        dropout_shape = list(attn_weights.shape)
        dropout_shape[-2] = 1
        keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
        keep = jnp.broadcast_to(keep, attn_weights.shape)
        multiplier = keep.astype(attn_weights.dtype) / jnp.asarray(
            keep_prob, dtype=dtype
        )
        attn_weights = attn_weights * multiplier

    # Take the linear combination of `value`.
    return jnp.einsum("bhqk,bkhd->bqhd", attn_weights, value)


dynamic_vector_slice_in_dim = jax.vmap(
    lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None)
)


class MultiHeadDotProductAttention(nn.Module):
    """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      head_dim: dimension of each head.
      dtype: the dtype of the computation.
      dropout_rate: dropout rate
      kernel_init: initializer for the kernel of the Dense layers.
      float32_logits: bool, if True then compute logits in float32 to avoid
        numerical issues with bfloat16.
    """

    num_heads: int
    dtype: DType = jnp.float32
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    dropout_rate: float = 0.0
    qkv_kernel_init: Initializer = nn.initializers.variance_scaling(
        1.0, "fan_in", "normal"
    )
    out_kernel_init: Initializer = nn.initializers.variance_scaling(
        1.0, "fan_in", "normal"
    )
    bias_init: Initializer = nn.initializers.zeros
    float32_logits: bool = False  # computes logits in float32 for stability.

    @nn.compact
    def __call__(
        self,
        inputs_q: Array,
        inputs_kv: Array,
        mask: Optional[Array] = None,
        bias: Optional[Array] = None,
        *,
        decode: bool = False,
        deterministic: bool = False,
    ) -> Array:
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        There are two modes: decoding and non-decoding (e.g., training). The mode is
        determined by `decode` argument. For decoding, this method is called twice,
        first to initialize the cache and then for an actual decoding process. The
        two calls are differentiated by the presence of 'cached_key' in the variable
        dict. In the cache initialization stage, the cache variables are initialized
        as zeros and will be filled in the subsequent decoding process.

        In the cache initialization call, `inputs_q` has a shape [batch, length,
        q_features] and `inputs_kv`: [batch, length, kv_features]. During the
        incremental decoding stage, query, key and value all have the shape [batch,
        1, qkv_features] corresponding to a single step.

        Args:
          inputs_q: input queries of shape `[batch, q_length, q_features]`.
          inputs_kv: key/values of shape `[batch, kv_length, kv_features]`.
          mask: attention mask of shape `[batch, num_heads, q_length, kv_length]`.
          bias: attention bias of shape `[batch, num_heads, q_length, kv_length]`.
          decode: Whether to prepare and use an autoregressive cache.
          deterministic: Disables dropout if set to True.

        Returns:
          output of shape `[batch, length, q_features]`.
        """
        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        assert (
            qkv_features % self.num_heads == 0
        ), "Memory dimension must be divisible by number of heads."
        head_dim = qkv_features // self.num_heads

        projection = functools.partial(
            DenseGeneral,
            axis=-1,
            features=(self.num_heads, head_dim),
            kernel_axes=("embed", "joined_kv"),
            dtype=self.dtype,
        )
        # kaiming: the origial version;
        # if we use this, the numerical results should be the same as original
        # projection = functools.partial(
        #     nn.DenseGeneral,
        #     axis=-1,
        #     features=(self.num_heads, head_dim),
        #     bias_init=self.bias_init,
        #     use_bias=True,
        #     precision=None)

        # Project inputs_q to multi-headed q/k/v
        # dimensions are then [batch, length, num_heads, head_dim]
        query = projection(kernel_init=self.qkv_kernel_init, name="query")(inputs_q)
        key = projection(kernel_init=self.qkv_kernel_init, name="key")(inputs_kv)
        value = projection(kernel_init=self.qkv_kernel_init, name="value")(inputs_kv)

        query = with_sharding_constraint(query, ("batch", "length", "heads", "kv"))
        key = with_sharding_constraint(key, ("batch", "length", "heads", "kv"))
        value = with_sharding_constraint(value, ("batch", "length", "heads", "kv"))

        if decode:
            raise NotImplementedError
            # Detect if we're initializing by absence of existing cache data.
            is_initialized = self.has_variable("cache", "cached_key")
            # The key and value have dimension [batch, length, num_heads, head_dim],
            # but we cache them as [batch, num_heads, head_dim, length] as a TPU
            # fusion optimization. This also enables the "scatter via one-hot
            # broadcast" trick, which means we do a one-hot broadcast instead of a
            # scatter/gather operations, resulting in a 3-4x speedup in practice.
            swap_dims = lambda x: x[:-3] + tuple(x[i] for i in [-2, -1, -3])
            cached_key = self.variable(
                "cache", "cached_key", jnp.zeros, swap_dims(key.shape), key.dtype
            )
            cached_value = self.variable(
                "cache", "cached_value", jnp.zeros, swap_dims(value.shape), value.dtype
            )
            cache_index = self.variable(
                "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
            )
            if is_initialized:
                batch, num_heads, head_dim, length = cached_key.value.shape
                # During fast autoregressive decoding, we feed one position at a time,
                # and cache the keys and values step by step.
                # Sanity shape check of cached key against input query.
                expected_shape = (batch, 1, num_heads, head_dim)
                if expected_shape != query.shape:
                    raise ValueError(
                        "Autoregressive cache shape error, "
                        "expected query shape %s instead got %s."
                        % (expected_shape, query.shape)
                    )

                # Create a OHE of the current index. NOTE: the index is increased below.
                cur_index = cache_index.value
                one_hot_indices = jax.nn.one_hot(cur_index, length, dtype=key.dtype)
                # In order to update the key, value caches with the current key and
                # value, we move the length axis to the back, similar to what we did for
                # the cached ones above.
                # Note these are currently the key and value of a single position, since
                # we feed one position at a time.
                one_token_key = jnp.moveaxis(key, -3, -1)
                one_token_value = jnp.moveaxis(value, -3, -1)
                # Update key, value caches with our new 1d spatial slices.
                # We implement an efficient scatter into the cache via one-hot
                # broadcast and addition.
                key = cached_key.value + one_token_key * one_hot_indices
                value = cached_value.value + one_token_value * one_hot_indices
                cached_key.value = key
                cached_value.value = value
                cache_index.value = cache_index.value + 1
                # Move the keys and values back to their original shapes.
                key = jnp.moveaxis(key, -1, -3)
                value = jnp.moveaxis(value, -1, -3)

                # Causal mask for cached decoder self-attention: our single query
                # position should only attend to those key positions that have already
                # been generated and cached, not the remaining zero elements.
                mask = combine_masks(
                    mask,
                    jnp.broadcast_to(
                        jnp.arange(length) <= cur_index,
                        # (1, 1, length) represent (head dim, query length, key length)
                        # query length is 1 because during decoding we deal with one
                        # index.
                        # The same mask is applied to all batch elements and heads.
                        (batch, 1, 1, length),
                    ),
                )

                # Grab the correct relative attention bias during decoding. This is
                # only required during single step decoding.
                if bias is not None:
                    # The bias is a full attention matrix, but during decoding we only
                    # have to take a slice of it.
                    # This is equivalent to bias[..., cur_index:cur_index+1, :].
                    bias = dynamic_vector_slice_in_dim(
                        jnp.squeeze(bias, axis=0), jnp.reshape(cur_index, (-1)), 1, -2
                    )

        # Convert the boolean attention mask to an attention bias.
        if mask is not None:
            # raise NotImplementedError
            # attention mask in the form of attention bias
            attention_bias = lax.select(
                mask > 0,
                jnp.full(mask.shape, 0.0).astype(self.dtype),
                jnp.full(mask.shape, -1e10).astype(self.dtype),
            )
        else:
            attention_bias = None

        # Add provided bias term (e.g. relative position embedding).
        if bias is not None:
            raise NotImplementedError
            attention_bias = combine_biases(attention_bias, bias)

        dropout_rng = None
        if not deterministic and self.dropout_rate > 0.0:
            dropout_rng = self.make_rng("dropout")

        # Apply attention.
        x = dot_product_attention(
            query,
            key,
            value,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
            dtype=self.dtype,
            float32_logits=self.float32_logits,
        )

        # Back to the original inputs dimensions.
        out = DenseGeneral(
            features=features,
            axis=(-2, -1),
            kernel_init=self.out_kernel_init,
            kernel_axes=("joined_kv", "embed"),
            dtype=self.dtype,
            name="out",
        )(x)
        return out


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
    # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
    return tuple([ax if ax >= 0 else ndim + ax for ax in axes])


def _canonicalize_tuple(x):
    if isinstance(x, Iterable):
        return tuple(x)
    else:
        return (x,)


# ------------------------------------------------------------------------------
# DenseGeneral for attention layers.
# ------------------------------------------------------------------------------
class DenseGeneral(nn.Module):
    """A linear transformation with flexible axes.

    Attributes:
      features: tuple with numbers of output features.
      axis: tuple with axes to apply the transformation on.
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
    """

    features: Union[Iterable[int], int]
    use_bias: bool = True
    dtype: DType = jnp.float32
    kernel_init: Initializer = nn.initializers.variance_scaling(
        1.0, "fan_in", "truncated_normal"
    )
    bias_init: Initializer = nn.initializers.zeros
    kernel_axes: Tuple[str, ...] = ()
    axis: Union[Iterable[int], int] = -1

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """Applies a linear transformation to the inputs along multiple dimensions.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        features = _canonicalize_tuple(self.features)
        axis = _canonicalize_tuple(self.axis)

        inputs = jnp.asarray(inputs, self.dtype)
        axis = _normalize_axes(axis, inputs.ndim)

        kernel_shape = tuple([inputs.shape[ax] for ax in axis]) + features
        kernel_param_shape = (
            np.prod([inputs.shape[ax] for ax in axis]),
            np.prod(features),
        )
        kernel = param_with_axes(
            "kernel",
            self.kernel_init,
            kernel_param_shape,
            jnp.float32,
            axes=self.kernel_axes,
        )
        kernel = jnp.asarray(kernel, self.dtype)
        kernel = jnp.reshape(kernel, kernel_shape)

        contract_ind = tuple(range(0, len(axis)))
        y = lax.dot_general(inputs, kernel, ((axis, contract_ind), ((), ())))

        if self.use_bias:
            bias = param_with_axes(
                "bias",
                self.bias_init,
                kernel_param_shape[-1],
                jnp.float32,
                axes=(self.kernel_axes[-1],),
            )
            bias = jnp.asarray(bias, self.dtype)
            bias = jnp.reshape(bias, (1,) * (y.ndim - len(features)) + features)
            y += bias
        return y


class Dense(nn.Module):
    """A linear transformation applied over the last dimension of the input.

    Attributes:
      features: the number of output features.
      use_bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: float32).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer function for the weight matrix.
      bias_init: initializer function for the bias.
    """

    features: int
    use_bias: bool = True
    dtype: DType = jnp.float32
    param_dtype: DType = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Callable[
        [PRNGKey, Shape, DType], Array
    ] = nn.linear.default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, DType], Array] = nn.initializers.zeros
    kernel_axes: Tuple[str, ...] = ()

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = param_with_axes(
            "kernel",
            self.kernel_init,
            (inputs.shape[-1], self.features),
            self.param_dtype,
            axes=self.kernel_axes,
        )
        kernel = jnp.asarray(kernel, self.dtype)
        y = lax.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if self.use_bias:
            bias = param_with_axes(
                "bias",
                self.bias_init,
                (self.features,),
                self.param_dtype,
                axes=(self.kernel_axes[-1],),
            )
            bias = jnp.asarray(bias, self.dtype)
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


# ------------------------------------------------------------------------------
# Conv layers
# ------------------------------------------------------------------------------
class Conv(nn.Conv):
    """Conv with axis names:
    Copied from flax.linen.linear, replace self.param
    """

    kernel_axes: Tuple[str, ...] = ()

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """Applies a (potentially unshared) convolution to the inputs.

        Args:
          inputs: input data with dimensions (batch, spatial_dims..., features).
            This is the channels-last convention, i.e. NHWC for a 2d convolution
            and NDHWC for a 3D convolution. Note: this is different from the input
            convention used by `lax.conv_general_dilated`, which puts the spatial
            dimensions last.

        Returns:
          The convolved data.
        """

        inputs = jnp.asarray(inputs, self.dtype)

        if isinstance(self.kernel_size, int):
            raise TypeError(
                "The kernel size must be specified as a"
                " tuple/list of integers (eg.: [3, 3])."
            )
        else:
            kernel_size = tuple(self.kernel_size)

        def maybe_broadcast(
            x: Optional[Union[int, Sequence[int]]]
        ) -> (Tuple[int, ...]):
            if x is None:
                # backward compatibility with using None as sentinel for
                # broadcast 1
                x = 1
            if isinstance(x, int):
                return (x,) * len(kernel_size)
            return tuple(x)

        is_single_input = False
        if inputs.ndim == len(kernel_size) + 1:
            is_single_input = True
            inputs = jnp.expand_dims(inputs, axis=0)

        # self.strides or (1,) * (inputs.ndim - 2)
        strides = maybe_broadcast(self.strides)
        input_dilation = maybe_broadcast(self.input_dilation)
        kernel_dilation = maybe_broadcast(self.kernel_dilation)

        padding_lax: Union[str, Sequence[Tuple[int, int]]]
        if self.padding == "CIRCULAR":
            kernel_size_dilated = [
                (k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)
            ]
            zero_pad: nn.linear.List[Tuple[int, int]] = [(0, 0)]
            pads = (
                zero_pad
                + [((k - 1) // 2, k // 2) for k in kernel_size_dilated]
                + [(0, 0)]
            )
            inputs = jnp.pad(inputs, pads, mode="wrap")
            padding_lax = "VALID"
        else:
            padding_lax = self.padding

        dimension_numbers = nn.linear._conv_dimension_numbers(inputs.shape)
        in_features = inputs.shape[-1]

        if self.shared_weights:
            # One shared convolutional kernel for all pixels in the output.
            assert in_features % self.feature_group_count == 0
            kernel_shape = kernel_size + (
                in_features // self.feature_group_count,
                self.features,
            )

        else:
            if self.feature_group_count != 1:
                raise NotImplementedError(
                    f"`lax.conv_general_dilated_local` does not support "
                    f"`feature_group_count != 1`, got `{self.feature_group_count}`."
                )

            # Need to know the spatial output shape of a standard convolution to
            # create the unshared convolution kernel.
            conv_output_shape = nn.linear.eval_shape(
                lambda lhs, rhs: lax.conv_general_dilated(  # pylint: disable=g-long-lambda
                    lhs=lhs,
                    rhs=rhs,
                    window_strides=strides,
                    padding=padding_lax,
                    dimension_numbers=dimension_numbers,
                ),
                inputs,
                nn.linear.ShapedArray(
                    kernel_size + (in_features, self.features), inputs.dtype
                ),
            ).shape

            # One (unshared) convolutional kernel per each pixel in the output.
            kernel_shape = conv_output_shape[1:-1] + (
                np.prod(kernel_size) * in_features,
                self.features,
            )

        kernel = param_with_axes(
            "kernel",
            self.kernel_init,
            kernel_shape,
            self.param_dtype,
            axes=self.kernel_axes,
        )
        # kernel = self.param('kernel', self.kernel_init, kernel_shape,
        #                     self.param_dtype)
        kernel = jnp.asarray(kernel, self.dtype)

        if self.shared_weights:
            y = lax.conv_general_dilated(
                inputs,
                kernel,
                strides,
                padding_lax,
                lhs_dilation=input_dilation,
                rhs_dilation=kernel_dilation,
                dimension_numbers=dimension_numbers,
                feature_group_count=self.feature_group_count,
                precision=self.precision,
            )
        else:
            y = lax.conv_general_dilated_local(
                lhs=inputs,
                rhs=kernel,
                window_strides=strides,
                padding=padding_lax,
                filter_shape=kernel_size,
                lhs_dilation=input_dilation,
                rhs_dilation=kernel_dilation,
                dimension_numbers=dimension_numbers,
                precision=self.precision,
            )

        if self.use_bias:
            if self.shared_weights:
                # One bias weight per output channel, shared between pixels.
                bias_shape = (self.features,)
            else:
                # One bias weight per output entry, unshared betwen pixels.
                bias_shape = y.shape[1:]

            # bias = self.param('bias', self.bias_init, bias_shape, self.param_dtype)
            bias = param_with_axes(
                "bias",
                self.bias_init,
                bias_shape,
                self.param_dtype,
                axes=(self.kernel_axes[-1],),
            )
            bias = jnp.asarray(bias, self.dtype)
            bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)
            y += bias

        if is_single_input:
            y = jnp.squeeze(y, axis=0)
        return y


# ------------------------------------------------------------------------------
# Normalization layers
# ------------------------------------------------------------------------------
class LayerNorm(nn.Module):
    """Layer normalization with axis names."""

    epsilon: float = 1e-6
    dtype: Any = jnp.float32
    param_dtype: DType = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    scale_init: Initializer = nn.initializers.ones
    bias_init: Initializer = nn.initializers.zeros
    axes: Tuple[str, ...] = ()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Applies layer normalization on the input."""
        reduction_axes = (-1,)
        feature_axes = (-1,)

        mean, var = nn.normalization._compute_stats(x, reduction_axes, None, None)

        return _normalize(
            self,
            x,
            mean,
            var,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
            self.axes,
        )


def _normalize(
    mdl: nn.Module,
    x: Array,
    mean: Array,
    var: Array,
    reduction_axes: Axes,
    feature_axes: Axes,
    dtype: DType,
    param_dtype: DType,
    epsilon: float,
    use_bias: bool,
    use_scale: bool,
    bias_init: Callable[[PRNGKey, Shape, DType], Array],
    scale_init: Callable[[PRNGKey, Shape, DType], Array],
    axes: Tuple[str, ...] = (),
):
    """ "Normalizes the input of a normalization layer and optionally applies a learned scale and bias.

    Arguments:
      mdl: Module to apply the normalization in (normalization params will reside
        in this module).
      x: The input.
      mean: Mean to use for normalization.
      var: Variance to use for normalization.
      reduction_axes: The axes in ``x`` to reduce.
      feature_axes: Axes containing features. A separate bias and scale is learned
        for each specified feature.
      dtype: Dtype of the returned result.
      param_dtype: Dtype of the parameters.
      epsilon: Normalization epsilon.
      use_bias: If true, add a bias term to the output.
      use_scale: If true, scale the output.
      bias_init: Initialization function for the bias term.
      scale_init: Initialization function for the scaling function.

    Returns:
      The normalized input.
    """
    reduction_axes = nn.normalization._canonicalize_axes(x.ndim, reduction_axes)
    feature_axes = nn.normalization._canonicalize_axes(x.ndim, feature_axes)
    stats_shape = list(x.shape)
    for axis in reduction_axes:
        stats_shape[axis] = 1
    mean = mean.reshape(stats_shape)
    var = var.reshape(stats_shape)
    feature_shape = [1] * x.ndim
    reduced_feature_shape = []
    for ax in feature_axes:
        feature_shape[ax] = x.shape[ax]
        reduced_feature_shape.append(x.shape[ax])
    y = x - mean
    mul = lax.rsqrt(var + epsilon)
    if use_scale:
        scale = param_with_axes(
            "scale", scale_init, reduced_feature_shape, param_dtype, axes=axes
        ).reshape(feature_shape)
        # scale = mdl.param('scale', scale_init, reduced_feature_shape,
        #                   param_dtype).reshape(feature_shape)
        mul *= scale
    y *= mul
    if use_bias:
        bias = param_with_axes(
            "bias", bias_init, reduced_feature_shape, param_dtype, axes=axes
        ).reshape(feature_shape)
        # bias = mdl.param('bias', bias_init, reduced_feature_shape,
        #                  param_dtype).reshape(feature_shape)
        y += bias
    return jnp.asarray(y, dtype)


class Embed(nn.Module):
    """A parameterized function from integers [0, n) to d-dimensional vectors.

    Attributes:
      num_embeddings: number of embeddings.
      features: number of feature dimensions for each embedding.
      dtype: the dtype of the embedding vectors (default: float32).
      embedding_init: embedding initializer.
      one_hot: performs the gather with a one-hot contraction rather than a true
        gather. This is currently needed for SPMD partitioning.
    """

    num_embeddings: int
    features: int
    cast_input_dtype: Optional[DType] = None
    dtype: DType = jnp.float32
    attend_dtype: Optional[DType] = None
    embedding_init: Initializer = default_embed_init
    one_hot: bool = False
    embedding: Array = dataclasses.field(init=False)
    axes: Tuple[str, ...] = ("vocab", "embed")

    def setup(self):
        self.embedding = param_with_axes(
            "embedding",
            self.embedding_init,
            (self.num_embeddings, self.features),
            jnp.float32,
            axes=self.axes,
        )

    def __call__(self, inputs: Array) -> Array:
        """Embeds the inputs along the last dimension.

        Args:
          inputs: input data, all dimensions are considered batch dimensions.

        Returns:
          Output which is embedded input data.  The output shape follows the input,
          with an additional `features` dimension appended.
        """
        if self.cast_input_dtype:
            inputs = inputs.astype(self.cast_input_dtype)
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError("Input type must be an integer or unsigned integer.")
        if self.one_hot:
            iota = lax.iota(jnp.int32, self.num_embeddings)
            one_hot = jnp.array(inputs[..., jnp.newaxis] == iota, dtype=self.dtype)
            output = jnp.dot(one_hot, jnp.asarray(self.embedding, self.dtype))
        else:
            output = jnp.asarray(self.embedding, self.dtype)[inputs]
            output = with_sharding_constraint(
                output, ("batch", "length", self.axes[-1])
            )
        return output

    def attend(self, query: Array) -> Array:
        """Attend over the embedding using a query array.

        Args:
          query: array with last dimension equal the feature depth `features` of the
            embedding.

        Returns:
          An array with final dim `num_embeddings` corresponding to the batched
          inner-product of the array of query vectors against each embedding.
          Commonly used for weight-sharing between embeddings and logit transform
          in NLP models.
        """
        dtype = self.attend_dtype if self.attend_dtype is not None else self.dtype
        return jnp.dot(query, jnp.asarray(self.embedding, dtype).T)
