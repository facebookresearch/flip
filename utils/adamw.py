from typing import Any, Callable, NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp
from matplotlib.cbook import flatten

from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import utils

from optax._src import transform
# from optax._src.transform import _update_moment, _bias_correction, ScaleByAdamState
from optax._src.transform import _bias_correction, ScaleByAdamState
from optax._src.wrappers import MaskedState


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
    flatten_params: Optional[bool] = False,
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
          b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype, flatten_params=flatten_params),
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
    flatten_params: Optional[bool] = False,
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

  def reshape_and_zeros_like(param):
    if len(param.shape) == 2:
      return jnp.zeros_like(param).reshape([-1, 1])  # hack it could be [-1, 2] or other
    else:
      return jnp.zeros_like(param)    

  def init_fn(params):
    if flatten_params:
      mu = jax.tree_map(reshape_and_zeros_like, params)
      nu = jax.tree_map(reshape_and_zeros_like, params)
    else:
      mu = jax.tree_map(  # First moment
          lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
      nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = _update_moment(updates, state.mu, b1, 1, flatten_params=flatten_params)
    nu = _update_moment(updates, state.nu, b2, 2, flatten_params=flatten_params)
    count_inc = numerics.safe_int32_increment(state.count)
    mu_hat = _bias_correction(mu, b1, count_inc)
    nu_hat = _bias_correction(nu, b2, count_inc)
    updates = jax.tree_map(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
    mu = utils.cast_tree(mu, mu_dtype)
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


# # ------------------------------------------
# # AdaRows optimizer
# # ------------------------------------------
# class ScaleByAdaRowsState(NamedTuple):
#   """State for the AdaRows algorithm."""
#   count: chex.Array  # shape=(), dtype=jnp.int32.
#   mu: base.Updates
#   nu: base.Updates


# def adarows(
#     learning_rate: ScalarOrSchedule,
#     b1: float = 0.9,
#     b2: float = 0.999,
#     eps: float = 1e-8,
#     eps_root: float = 0.0,
#     mu_dtype: Optional[Any] = None,
#     weight_decay: float = 1e-4,
#     mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
# ) -> base.GradientTransformation:
#   """Kaiming: experimental AdamW per row

#   Returns:
#     the corresponding `GradientTransformation`.
#   """
#   return combine.chain(
#       _scale_by_adam_per_rows(
#           b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
#       transform.add_decayed_weights(weight_decay, mask),
#       _scale_by_learning_rate(learning_rate),
#   )


# def PartitionRuleScaleByAdaRowsState(state, params_axes):
#   nu_params_axes = jax.tree_map(lambda g: g if len(g) != 2 else type(g)('_null0',), params_axes)
#   return ScaleByAdaRowsState(count=None, mu=params_axes, nu=nu_params_axes)


# def _scale_by_adam_per_rows(
#     b1: float = 0.9,
#     b2: float = 0.999,
#     eps: float = 1e-8,
#     eps_root: float = 0.0,
#     mu_dtype: Optional[Any] = None,
# ) -> base.GradientTransformation:
#   """Rescale updates per row according to the Adam algorithm.

#   Returns:
#     An (init_fn, update_fn) tuple.
#   """

#   mu_dtype = utils.canonicalize_dtype(mu_dtype)
#   nu_dtype = utils.canonicalize_dtype(jnp.float32)

#   # e.g., in_channel=1024, out_channel=4096
#   # kernel: (1024, 4096)
#   # nu is reduced over axis=4096, and nu.shape=(1024,)
#   axis_col = 0
#   axis_row = 1

#   def init_fn(params):
#     mu = jax.tree_map(lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)  # First moment

#     nu = jax.tree_map(lambda t: jnp.zeros_like(
#       t if t.ndim != 2 else jnp.mean(t, axis=axis_row, keepdims=True),
#       dtype=nu_dtype), params)  # Second moment

#     return ScaleByAdaRowsState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

#   def update_fn(updates, state, params=None):
#     del params
#     mu = _update_moment(updates, state.mu, b1, 1)  # no change
#     # nu = _update_moment(updates, state.nu, b2, 2)  # original adam

#     nu = jax.tree_map(lambda g: g**2 if g.ndim != 2 else jnp.mean(g**2, axis=axis_row, keepdims=True), updates)
#     nu = jax.tree_map(lambda g, t: (1 - b2) * g + b2 * t, nu, state.nu)

#     count_inc = numerics.safe_int32_increment(state.count)
#     mu_hat = _bias_correction(mu, b1, count_inc)
#     nu_hat = _bias_correction(nu, b2, count_inc)
#     updates = jax.tree_map(
#         lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
#     mu = utils.cast_tree(mu, mu_dtype)
#     return updates, ScaleByAdaRowsState(count=count_inc, mu=mu, nu=nu)

#   return base.GradientTransformation(init_fn, update_fn)


def _update_moment(updates, moments, decay, order, flatten_params=False):
  """Compute the exponential moving average of the `order`-th moment."""
  if flatten_params:
      return jax.tree_map(
        lambda g, t: (1 - decay) * (g ** order) + decay * t.reshape(g.shape), updates, moments)
  else:
    return jax.tree_map(
        lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments)


# ------------------------------------------
# Masked optimizer
# ------------------------------------------
import optax
def masked(
    inner: base.GradientTransformation,
    mask: Union[base.PyTree, Callable[[base.Params], base.PyTree]]
) -> base.GradientTransformation:
  """Mask updates so only some are transformed, the rest are passed through.
  For example, it is common to skip weight decay for BatchNorm scale and all
  bias parameters. In many networks, these are the only parameters with only
  one dimension. So, you may create a mask function to mask these out as
  follows::
    mask_fn = lambda p: jax.tree_map(lambda x: x.ndim != 1, p)
    weight_decay = optax.masked(optax.add_decayed_weights(0.001), mask_fn)
  You may alternatively create the mask pytree upfront::
    mask = jax.tree_map(lambda x: x.ndim != 1, params)
    weight_decay = optax.masked(optax.add_decayed_weights(0.001), mask)
  For the ``inner`` transform, state will only be stored for the parameters that
  have a mask value of ``True``.
  Args:
    inner: Inner transformation to mask.
    mask: a PyTree with same structure as (or a prefix of) the params PyTree, or
      a Callable that returns such a pytree given the params/updates. The leaves
      should be booleans, ``True`` for leaves/subtrees you want to apply the
      transformation to, and ``False`` for those you want to skip. The mask must
      be static for the gradient transformation to be jit-compilable.
  Returns:
    New GradientTransformation wrapping ``inner``.
  """
  def mask_pytree(pytree, mask_tree):
    return jax.tree_map(lambda m, p: p if m else None, mask_tree, pytree)

  def init_fn(params):
    mask_tree = mask(params) if callable(mask) else mask
    masked_params = mask_pytree(params, mask_tree)
    return MaskedState(inner_state=inner.init(masked_params))

  def update_fn(updates, state, params=None):
    mask_tree = mask(updates) if callable(mask) else mask
    masked_updates = mask_pytree(updates, mask_tree)
    masked_params = None if params is None else mask_pytree(params, mask_tree)

    new_masked_updates, new_inner_state = inner.update(
        masked_updates, state.inner_state, masked_params)

    # new_updates = jax.tree_map(
    #     lambda m, new_u, old_u: new_u if m else old_u,
    #     mask_tree, new_masked_updates, updates)  # this takes the old updates
    new_updates = jax.tree_map(
        lambda m, new_u: new_u if m else 0.,
        mask_tree, new_masked_updates)  # this takes zero
    return new_updates, MaskedState(inner_state=new_inner_state)

  return base.GradientTransformation(init_fn, update_fn)