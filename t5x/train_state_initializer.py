
from absl import logging
import flax
from flax import struct
# from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, Type, Union

import t5x.train_state as train_state_lib
import t5x.optimizers

from utils import opt_util
from utils import lrd_util
from utils import adamw_util


Array = Union[np.ndarray, jnp.ndarray, jax.pxla.ShardedDeviceArray]


class TrainState(flax.training.train_state.TrainState):
  rng: Any
  variables: flax.core.FrozenDict[str, Any]
  ema_tx: optax.GradientTransformation = struct.field(pytree_node=False)
  ema_state: optax.EmaState


def initialized(rng, image_size, model, init_backend='tpu'):
  input_shape = (1, image_size, image_size, 3)
  def init(*args):
    return model.init(*args, train=False)
  init = jax.jit(init, backend=init_backend)
  logging.info('Initializing params...')
  variables = init({'params': rng}, jnp.ones(input_shape, model.dtype))
  logging.info('Initializing params done.')
  return variables


def init_fn(rng, image_size, model, init_backend='tpu'):
  input_shape = (1, image_size, image_size, 3)
  def init(*args):
    return model.init(*args, train=False)
  init = jax.jit(init, backend=init_backend)
  variables = init({'params': rng}, jnp.ones(input_shape, model.dtype))
  return variables


def initialized_shapes(rng, image_size, model):
  input_shape = (1, image_size, image_size, 3)
  def init(*args):
    return model.init(*args, train=False)
  variables_shape = jax.eval_shape(init, {'params': rng}, jnp.ones(input_shape, model.dtype))
  logging.info('variables_shape:\n{}'.format(variables_shape))
  return variables_shape


def create_optimizer(config, params_names, learning_rate_fn):
  # optional: exclude some wd
  mask = None
  if config.exclude_wd:
    mask = jax.tree_util.tree_map(lambda x, y: bool(x and y), 
      opt_util.filter_parameters(params_names, opt_util.filter_bias_and_norm),
      opt_util.filter_parameters(params_names, opt_util.filter_cls_and_posembed)
    )
  # logging.info('Apply wd: {}'.format(mask))

  optimizer_def = getattr(adamw_util, config.opt_type)  # optax.adamw
  optimizer_def = t5x.optimizers.wrap_optax_optimizer(optimizer_def)
  tx = optimizer_def(learning_rate=learning_rate_fn, **config.opt, mask=mask, mu_dtype=getattr(jnp, config.opt_mu_dtype))
  return tx

  if config.learning_rate_decay < 1.:
    raise NotImplementedError
    lrd_func = lrd_util.lrd_func(config.model.transformer.num_layers, config.learning_rate_decay)
    lrd = lrd_util.filter_parameters(params, lrd_func)
    # logging.info('Apply lrd: {}'.format(lrd))
    tx = optax._src.combine.chain(tx, lrd_util.scale_by_lrd(lrd))

  tx = optax.GradientTransformation(init=jax.jit(tx.init, backend=config.init_backend), update=tx.update)  # put to cpu
  return tx


def create_train_state(rng, config: ml_collections.ConfigDict,
                       model, image_size, learning_rate_fn):
  return __create_train_state(rng, config, model, image_size, learning_rate_fn)


def __create_train_state(rng, config: ml_collections.ConfigDict,
                       model, image_size, learning_rate_fn):
  """Create initial training state."""
  # create optimizer first
  params_shapes = initialized_shapes(jax.random.PRNGKey(0), image_size, model)  # inference names
  optimizer_def = create_optimizer(config, params_shapes['params'], learning_rate_fn)

  # ---------------------------------------------------------------------------
  def initialize_train_state(rng: Array):
    # split rng for init and for state
    rng_init, rng_state = jax.random.split(rng)
    initial_variables = init_fn(rng=rng_init, image_size=image_size, model=model)
    if optimizer_def:
      return train_state_lib.FlaxOptimTrainState.create(
          optimizer_def, initial_variables, rng=rng_state)
    return train_state_lib.InferenceState.create(initial_variables)

  state = initialize_train_state(rng)
  # train_state_shapes = jax.eval_shape(initialize_train_state, rng_init)
  return state
  # ---------------------------------------------------------------------------
  
  raise NotImplementedError

  variables = initialized(rng_init, image_size, model, config.init_backend)
  variables_states, params = variables.pop('params')

  # optional: rescale
  assert not config.rescale_init  # TODO: move to model
  # if config.rescale_init:
  #   rescales = opt_util.filter_parameters(params, opt_util.layer_rescale)
  #   params = jax.tree_util.tree_map(lambda x, y: x * y, rescales, params)

  # if config.rescale_head_init != 1.:
  #   params = flax.core.frozen_dict.unfreeze(params)
  #   params['head']['kernel'] *= config.rescale_head_init
  #   params = flax.core.frozen_dict.freeze(params)

  stds = jax.tree_util.tree_map(lambda x: np.array(x).std(), params)
  logging.info('std: {}'.format(stds))

  if config.ema:
    raise NotImplementedError
    ema_tx = optax.ema(decay=config.ema_decay, debias=False)
    ema_state = ema_tx.init(flax.core.frozen_dict.FrozenDict({'params': params, **variables_states}))
  else:
    ema_tx = None
    ema_state = None
  state = TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=tx,
      rng=rng_state,
      variables=variables_states,
      ema_tx=ema_tx,
      ema_state=ema_state)
  return state

