
from absl import logging
import flax
from flax import struct
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


# def initialized(rng, image_size, model, init_backend='tpu'):
#   input_shape = (1, image_size, image_size, 3)
#   def init(*args):
#     return model.init(*args, train=False)
#   init = jax.jit(init, backend=init_backend)
#   logging.info('Initializing params...')
#   variables = init({'params': rng}, jnp.ones(input_shape, model.dtype))
#   logging.info('Initializing params done.')
#   return variables


def init_fn(rng, input_shape, model, init_backend='tpu'):
  # input_shape = (1, image_size, image_size, 3)
  def init(*args):
    return model.init(*args, train=False)
  init = jax.jit(init, backend=init_backend)
  variables = init({'params': rng}, jnp.ones(input_shape, model.dtype))
  return variables


def initialized_shapes(rng, input_shape, model):
  # input_shape = (1, image_size, image_size, 3)
  def init(*args):
    return model.init(*args, train=False)
  variables_shape = jax.eval_shape(init, {'params': rng}, jnp.ones(input_shape, model.dtype))
  return variables_shape


def create_optimizer(config, params_names=None, learning_rate_fn=None):
  # optional: exclude some wd
  mask = None
  if config.exclude_wd and params_names is not None:
    raise NotImplementedError
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
                       model_wrapped, image_size, learning_rate_fn, partitioner):
  """Create initial training state."""
  input_shape = (config.batch_size, image_size, image_size, 3)
  # input_shape = (config.batch_size, image_size // config.model.patches.size[0] * image_size // config.model.patches.size[1], 4)
  # create optimizer first
  # params_shapes = initialized_shapes(jax.random.PRNGKey(0), input_shape, model_wrapped.module)  # inference names
  # optimizer_def = create_optimizer(config, params_shapes['params'], learning_rate_fn)
  optimizer_def = model_wrapped.optimizer_def

  # optional: rescale
  assert not config.rescale_init  # TODO: move to model

  # ---------------------------------------------------------------------------
  def initialize_train_state(rng: Array):
    # split rng for init and for state
    rng_init, rng_state = jax.random.split(rng)
    initial_variables = init_fn(rng=rng_init, input_shape=input_shape, model=model_wrapped.module)
    if optimizer_def:
      return train_state_lib.FlaxOptimTrainState.create(
          optimizer_def, initial_variables, rng=rng_state)
    return train_state_lib.InferenceState.create(initial_variables)

  global_train_state_shape = jax.eval_shape(initialize_train_state, rng=rng)
  train_state_axes = partitioner.get_mesh_axes(global_train_state_shape)

  p_initialize_train_state_fn = partitioner.partition(
      initialize_train_state,
      in_axis_resources=None,
      out_axis_resources=train_state_axes)

  logging.info('Initializing train_state...')
  train_state = p_initialize_train_state_fn(rng)
  logging.info('Initializing train_state done.')

  # for debug
  # k = train_state.params['Transformer']['encoderblock_00']['MlpBlock_0']['Dense_0']['kernel']
  # k.sharding_spec

  # --------------------------------------------------
  # not partitioned
  # --------------------------------------------------
  # train_state = initialize_train_state(rng)
  return train_state, train_state_axes, global_train_state_shape
  

