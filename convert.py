"""
Convert PyTorch models to JAX for debugging
"""

import functools
import time, datetime
from typing import Any

from absl import logging
from clu import metric_writers
import flax
from flax import struct
from flax.training import checkpoints
import jax
import jax.numpy as jnp
from jax import random
import ml_collections

import models_vit

from utils import summary_util as summary_util  # must be after 'from clu import metric_writers'
from utils import convert_util


import jax.profiler

import numpy as np
import os


NUM_CLASSES = 1000


def create_model(*, model_cls, half_precision, **kwargs):
  assert not half_precision
  return model_cls(num_classes=NUM_CLASSES, **kwargs)


def initialized(key, image_size, model, init_backend='tpu'):
  input_shape = (1, image_size, image_size, 3)
  def init(*args):
    return model.init(*args, train=False)
  init = jax.jit(init, backend=init_backend)
  variables = init({'params': key}, jnp.ones(input_shape, model.dtype))
  return variables


class TrainState(struct.PyTreeNode):
  params: flax.core.FrozenDict[str, Any]


def create_train_state(rng, config: ml_collections.ConfigDict,
                       model, image_size):
  """Create initial training state."""

  # split rng for init and for state
  rng_init, rng_state = jax.random.split(rng)

  variables = initialized(rng_init, image_size, model, config.init_backend)
  variables_states, params = variables.pop('params')

  state = TrainState(params=params)
  return state


def convert_model(config: ml_collections.ConfigDict, workdir: str, mode: str) -> TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """
  start_time = time.time()
  assert config.pretrain_dir != ''

  writer = metric_writers.create_default_writer(
      logdir=workdir, just_logging=jax.process_index() != 0)

  rng = random.PRNGKey(0)

  image_size = 224

  model_cls = models_vit.VisionTransformer
  model = create_model(model_cls=model_cls, half_precision=config.half_precision, **config.model)

  logging.info('Creating TrainState:')
  state = create_train_state(rng, config, model, image_size)

  if mode == 'p2j':
    logging.info('Converting from PyTorch checkpoints to JAX:')
    convert_util.convert_from_pytorch(state, config.pretrain_dir)
  elif mode == 'j2p':
    logging.info('Converting from JAX checkpoints to PyTorch:')
    convert_util.convert_to_pytorch(state, config.pretrain_dir, config)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  total_time = time.time() - start_time
  total_time_str = str(datetime.timedelta(seconds=int(total_time)))
  logging.info('Elapsed time: {}'.format(total_time_str))

  return state
