from typing import Any, Callable

from jax import random
from jax import core

import jax.numpy as jnp
from jax import dtypes


DType = Any
def constant(value, dtype: DType = jnp.float_) -> Callable:
  """Builds an initializer that returns arrays full of a constant ``value``.

  Args:
    value: the constant value with which to fill the initializer.
    dtype: optional; the initializer's default dtype.

  >>> import jax, jax.numpy as jnp
  >>> initializer = jax.nn.initializers.constant(-7)
  >>> initializer(jax.random.PRNGKey(42), (2, 3), jnp.float32)
  DeviceArray([[-7., -7., -7.],
               [-7., -7., -7.]], dtype=float32)
  """
  def init(key, shape, dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    return jnp.full(shape, value, dtype=dtype)
  return init


def patch_kernel(dtype: DType = jnp.float_):
  """
  ViT patch embedding initializer:
  As patch_embed is implemented as Conv, we view its 4D params as 2D
  """
  def init(key, shape, dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    h, w, c, n = shape
    fan_in = h * w * c
    fan_out = n
    denominator = (fan_in + fan_out) / 2
    variance = jnp.array(1. / denominator, dtype=dtype)
    return random.uniform(key, shape, dtype, -1) * jnp.sqrt(3 * variance)

  return init