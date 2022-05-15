
from absl import logging
import flax
from flax import struct
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import optax

from typing import Any


from utils import opt_util
from utils import lrd_util
from utils import adamw_util


class TrainState(train_state.TrainState):
  rng: Any
  variables: flax.core.FrozenDict[str, Any]
  ema_tx: optax.GradientTransformation = struct.field(pytree_node=False)
  ema_state: optax.EmaState


def initialized(key, image_size, model, init_backend='tpu'):
  input_shape = (1, image_size, image_size, 3)
  def init(*args):
    return model.init(*args, train=False)
  init = jax.jit(init, backend=init_backend)
  logging.info('Initializing params...')
  variables = init({'params': key}, jnp.ones(input_shape, model.dtype))
  logging.info('Initializing params done.')

  variables_shape = jax.eval_shape(init, {'params': key}, jnp.ones(input_shape, model.dtype))
  logging.info('variables_shape:\n{}'.format(variables_shape))

  return variables


def create_train_state(rng, config: ml_collections.ConfigDict,
                       model, image_size, learning_rate_fn):
  """Create initial training state."""
  # split rng for init and for state
  rng_init, rng_state = jax.random.split(rng)

  variables = initialized(rng_init, image_size, model, config.init_backend)
  variables_states, params = variables.pop('params')

  # optional: rescale
  if config.rescale_init:
    rescales = opt_util.filter_parameters(params, opt_util.layer_rescale)
    params = jax.tree_util.tree_map(lambda x, y: x * y, rescales, params)

  if config.rescale_head_init != 1.:
    params = flax.core.frozen_dict.unfreeze(params)
    params['head']['kernel'] *= config.rescale_head_init
    params = flax.core.frozen_dict.freeze(params)

  # stds = jax.tree_util.tree_map(lambda x: np.array(x).std(), params)
  # logging.info('std: {}'.format(stds))

  # optional: exclude some wd
  if config.exclude_wd:
    mask = jax.tree_util.tree_map(lambda x, y: bool(x and y), 
      opt_util.filter_parameters(params, opt_util.filter_bias_and_norm),
      opt_util.filter_parameters(params, opt_util.filter_cls_and_posembed)
    )
  else:
    mask = None
  # logging.info('Apply weight decay: {}'.format(mask))

  # tx = getattr(optax, config.opt_type)  # optax.adamw
  tx = getattr(adamw_util, config.opt_type)  # optax.adamw
  tx = tx(learning_rate=learning_rate_fn, **config.opt, mask=mask, mu_dtype=getattr(jnp, config.opt_mu_dtype))

  if config.learning_rate_decay < 1.:
    lrd_func = lrd_util.lrd_func(config.model.transformer.num_layers, config.learning_rate_decay)
    lrd = lrd_util.filter_parameters(params, lrd_func)
    # logging.info('Apply lrd: {}'.format(lrd))
    tx = optax._src.combine.chain(tx, lrd_util.scale_by_lrd(lrd))

  tx = optax.GradientTransformation(init=jax.jit(tx.init, backend=config.init_backend), update=tx.update)  # put to cpu

  if config.ema:
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

