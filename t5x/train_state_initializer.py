
from absl import logging
import jax
import jax.numpy as jnp
import optax
import functools

import t5x.train_state as train_state_lib
import t5x.optimizers

from utils import opt_util
from utils import lrd_util
from utils import adamw


def init_fn(rng, image_size, model):
  input_shape = (1, image_size, image_size, 3)
  variables = model.init({'params': rng, 'dropout': jax.random.PRNGKey(0)}, jnp.ones(input_shape, model.dtype), train=True)
  return variables


def init_shapes(rng, image_size, model):
  input_shape = (1, image_size, image_size, 3)
  init = functools.partial(model.init, train=True) 
  variables_shape = jax.eval_shape(init, {'params': rng, 'dropout': jax.random.PRNGKey(0)}, jnp.ones(input_shape, model.dtype))
  return variables_shape


def create_learning_rate_fn(
    config,
    base_learning_rate: float,
    steps_per_epoch: int):
  """Create learning rate schedule."""
  warmup_fn = optax.linear_schedule(
      init_value=config.warmup_abs_lr, end_value=base_learning_rate,
      transition_steps=config.warmup_epochs * steps_per_epoch)
  cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
  cosine_fn = optax.cosine_decay_schedule(
      init_value=base_learning_rate,
      decay_steps=cosine_epochs * steps_per_epoch,
      alpha=config.min_abs_lr / base_learning_rate)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[config.warmup_epochs * steps_per_epoch])
  return schedule_fn


def create_optimizer(config, params_names, steps_per_epoch):

  # create the lr schedule function
  abs_learning_rate = config.learning_rate * config.batch_size / 256.
  learning_rate_fn = create_learning_rate_fn(config, abs_learning_rate, steps_per_epoch)

  if config.opt_type in {'adamw', 'adarows'}:
    # optional: exclude some wd
    mask = None
    if config.exclude_wd:
      mask = jax.tree_util.tree_map(lambda x, y: bool(x and y), 
        opt_util.filter_parameters(params_names, opt_util.filter_bias_and_norm),
        opt_util.filter_parameters(params_names, opt_util.filter_cls_and_posembed)
      )
    # logging.info('Apply wd: {}'.format(mask))

    opt = getattr(adamw, config.opt_type)  # optax.adamw
    opt = t5x.optimizers.wrap_optax_optimizer(opt)
    opt = opt(learning_rate=learning_rate_fn, **config.opt, mask=mask, mu_dtype=getattr(jnp, config.opt_mu_dtype))
    opt.metric_learning_rate_fn = learning_rate_fn  # hack for metric

    if config.learning_rate_decay < 1.:
      lrd_func = lrd_util.lrd_func(config.model.transformer.num_layers, config.learning_rate_decay)
      lrd = lrd_util.filter_parameters(params_names, lrd_func)
      # logging.info('Apply lrd: {}'.format(lrd))
      opt.optax_optimizer = optax._src.combine.chain(opt.optax_optimizer, lrd_util.scale_by_lrd(lrd))
  else:
    raise NotImplementedError

  return opt


def create_train_state(config, model, image_size, steps_per_epoch, partitioner):
  """Create initial training state."""
  rng = jax.random.PRNGKey(0)  # for shape reference only
  # create optimizer first
  params_shapes = init_shapes(rng, image_size, model)
  opt = create_optimizer(config, params_shapes['params'], steps_per_epoch)

  # ---------------------------------------------------------------------------
  def initialize_train_state(rng_init):
    # split rng for init and for state
    initial_variables = init_fn(rng=rng_init, image_size=image_size, model=model)
    if opt:
      return train_state_lib.FlaxOptimTrainState.create(opt, initial_variables)
    return train_state_lib.InferenceState.create(initial_variables)
  train_state_shape = jax.eval_shape(initialize_train_state, rng_init=rng)
  train_state_axes = partitioner.get_mesh_axes(train_state_shape)

  p_init_fn = partitioner.partition(
      initialize_train_state,
      in_axis_resources=None,
      out_axis_resources=train_state_axes)

  return p_init_fn, train_state_axes, train_state_shape
  

