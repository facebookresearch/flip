from typing import Any, Callable, NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp

from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import utils

from optax._src import transform
from optax._src.transform import _update_moment, _bias_correction, ScaleByAdamState



ScalarOrSchedule = Union[float, base.Schedule]


def _scale_by_learning_rate(learning_rate: ScalarOrSchedule, flip_sign=True):
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return transform.scale_by_schedule(lambda count: m * learning_rate(count))
  return transform.scale(m * learning_rate)


# ------------------------------------------
# AdamW optimizer, revised for data type
# ------------------------------------------
def adamw(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
  """Adam with weight decay regularization.

  AdamW uses weight decay to regularise learning towards small weights, as
  this leads to better generalisation. In SGD you can also use L2 regularisation
  to implement this as an additive loss term, however L2 regularization
  does not behave as intended for adaptive gradient algorithms such as Adam.

  WARNING: Sometimes you may want to skip weight decay for BatchNorm scale or
  for the bias parameters. You can use `optax.masked` to make your own AdamW
  variant where `additive_weight_decay` is applied only to a subset of `params`.

  References:
    Loshchilov et al, 2019: https://arxiv.org/abs/1711.05101

  Args:
    learning_rate: this is a fixed global scaling factor.
    b1: the exponential decay rate to track the first moment of past gradients.
    b2: the exponential decay rate to track the second moment of past gradients.
    eps: a small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: (default `0`), a small constant applied to denominator inside the
      square root (as in RMSProp), to avoid dividing by zero when rescaling.
      This is needed for instance when computing (meta-)gradients through Adam.
    mu_dtype: optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype is inferred from `params` and `updates`.
    weight_decay: strength of the weight decay regularization.
    mask: a tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the transformation to, and `False` for those you want to skip.

  Returns:
    the corresponding `GradientTransformation`.
  """
  return combine.chain(
      _scale_by_adam(
          b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
      transform.add_decayed_weights(weight_decay, mask),
      _scale_by_learning_rate(learning_rate),
  )


# from optax.transform
def _scale_by_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  """Rescale updates according to the Adam algorithm.

  References:
    [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)

  Args:
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    mu_dtype: optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype is inferred from `params` and `updates`.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = jax.tree_map(  # First moment
        lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
    nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = _update_moment(updates, state.mu, b1, 1)
    nu = _update_moment(updates, state.nu, b2, 2)
    count_inc = numerics.safe_int32_increment(state.count)
    mu_hat = utils.cast_tree(_bias_correction(mu, b1, count_inc), mu_dtype)
    nu_hat = _bias_correction(nu, b2, count_inc)
    updates = jax.tree_map(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
    mu = utils.cast_tree(mu, mu_dtype)
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


# ------------------------------------------
# AdaRows optimizer
# ------------------------------------------
class ScaleByAdaRowsState(NamedTuple):
  """State for the AdaRows algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  nu: base.Updates


def adarows(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
  """Kaiming: experimental AdamW per row

  Returns:
    the corresponding `GradientTransformation`.
  """
  return combine.chain(
      _scale_by_adam_per_rows(
          b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
      transform.add_decayed_weights(weight_decay, mask),
      _scale_by_learning_rate(learning_rate),
  )


def PartitionRuleScaleByAdaRowsState(state, params_axes):
  nu_params_axes = jax.tree_map(lambda a: a if len(a) != 2 else type(a)('_null0',), params_axes)
  return ScaleByAdaRowsState(count=None, mu=params_axes, nu=nu_params_axes)


def _scale_by_adam_per_rows(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  """Rescale updates according to the Adam algorithm.

  References:
    [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)

  Args:
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    mu_dtype: optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype is inferred from `params` and `updates`.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  mu_dtype = utils.canonicalize_dtype(mu_dtype)
  nu_dtype = utils.canonicalize_dtype(jnp.float32)

  # e.g., in_channel=1024, out_channel=4096
  # kernel: (1024, 4096)
  # nu is reduced over axis=4096, and nu.shape=(1024,)
  axis_col = 0
  axis_row = 1

  def init_fn(params):
    mu = jax.tree_map(lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)  # First moment

    nu = jax.tree_map(lambda t: jnp.zeros_like(
      t if len(t.shape) != 2 else jnp.mean(t, axis=axis_row),
      dtype=nu_dtype), params)  # Second moment
    # nu = jax.tree_map(lambda t: jnp.zeros_like(t, dtype=nu_dtype), params)  # Second moment

    return ScaleByAdaRowsState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = _update_moment(updates, state.mu, b1, 1)
    nu = _update_moment(updates, state.nu, b2, 2)
    count_inc = numerics.safe_int32_increment(state.count)
    mu_hat = utils.cast_tree(_bias_correction(mu, b1, count_inc), mu_dtype)
    nu_hat = _bias_correction(nu, b2, count_inc)
    updates = jax.tree_map(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
    mu = utils.cast_tree(mu, mu_dtype)
    return updates, ScaleByAdaRowsState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)