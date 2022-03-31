# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ImageNet example.

This script trains a ViT on the ImageNet dataset.
The data is loaded using tensorflow_datasets.
"""

import functools
import time
from typing import Any

from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax
from flax import jax_utils
from flax import optim
from flax import struct
from flax.training import checkpoints
from flax.training import common_utils
# from flax.training import train_state
import utils.train_state as train_state
import utils.adamw_util as adamw_util
import jax
from jax import lax
import jax.numpy as jnp
from jax import random
import ml_collections
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

import input_pipeline
import models_vit

from utils import summary_util as summary_util  # must be after 'from clu import metric_writers'
from utils import opt_util
from utils import mix_util
from utils import checkpoint_util


import jax.profiler

import numpy as np
import os


NUM_CLASSES = 1000


def create_model(*, model_cls, half_precision, **kwargs):
  assert not half_precision
  # platform = jax.local_devices()[0].platform
  # if half_precision:
  #   if platform == 'tpu':
  #     model_dtype = jnp.bfloat16
  #   else:
  #     model_dtype = jnp.float16
  # else:
  #   model_dtype = jnp.float32
  return model_cls(num_classes=NUM_CLASSES, **kwargs)


def initialized(key, image_size, model, init_backend='tpu'):
  input_shape = (1, image_size, image_size, 3)
  def init(*args):
    return model.init(*args, train=False)
  init = jax.jit(init, backend=init_backend)
  variables = init({'params': key}, jnp.ones(input_shape, model.dtype))
  return variables


def cross_entropy_loss(logits, labels_one_hot):
  xentropy = optax.softmax_cross_entropy(logits=logits, labels=labels_one_hot)
  return jnp.mean(xentropy)


def compute_metrics(logits, labels, labels_one_hot):
  loss = cross_entropy_loss(logits, labels_one_hot)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  metrics = lax.pmean(metrics, axis_name='batch')
  return metrics


def create_learning_rate_fn(
    config: ml_collections.ConfigDict,
    base_learning_rate: float,
    steps_per_epoch: int):
  """Create learning rate schedule."""
  warmup_fn = optax.linear_schedule(
      init_value=0., end_value=base_learning_rate,
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


def train_step(state, batch, learning_rate_fn, config):
  """Perform a single training step."""
  _, new_rng = jax.random.split(state.rng)
  # Bind the rng key to the device id (which is unique across hosts)
  # Note: This is only used for multi-host training (i.e. multiple computers
  # each with multiple accelerators).
  dropout_rng = jax.random.fold_in(state.rng, jax.lax.axis_index('batch'))
  def loss_fn(params):
    """loss function used for training."""
    mutable = [k for k in state.variables]
    outcome = state.apply_fn(
        {'params': params, **state.variables},
        inputs=batch['image'],
        mutable=mutable,
        rngs=dict(dropout=dropout_rng),
        train=True)
    logits, new_variables = outcome

    loss = cross_entropy_loss(logits, batch['label_one_hot'])
    return loss, (new_variables, logits)

  step = state.step
  dynamic_scale = state.dynamic_scale
  lr = learning_rate_fn(step)

  if dynamic_scale:
    grad_fn = dynamic_scale.value_and_grad(
        loss_fn, has_aux=True, axis_name='batch')
    dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
    # dynamic loss takes care of averaging gradients across replicas
  else:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grads = lax.pmean(grads, axis_name='batch')
  new_variables, logits = aux[1]
  metrics = compute_metrics(logits, batch['label'], batch['label_one_hot'])
  metrics['learning_rate'] = lr

  # ----------------------------------------------------------------------------
  # original
  # new_state = state.apply_gradients(grads=grads, variables=new_variables, rng=new_rng)
  # if new_state.ema is not None:
  #   new_ema = new_state.ema.update(flax.core.FrozenDict({'params': new_state.params, **new_variables}))
  #   new_state = new_state.replace(ema=new_ema)
  # ----------------------------------------------------------------------------

  # ----------------------------------------------------------------------------
  # modified impl.
  updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
  new_params = optax.apply_updates(state.params, updates)

  if config.ema:
    _, new_ema_state = state.ema_tx.update(
      updates=flax.core.frozen_dict.FrozenDict({'params': new_params, **new_variables}),
      state=state.ema_state)
  else:
    new_ema_state = None
  
  # new_ema = state.ema.update(flax.core.FrozenDict({'params': new_params, **new_variables})) if state.ema is not None else None
  new_state = state.replace(
    step=state.step + 1,
    params=new_params,
    opt_state=new_opt_state,
    variables=new_variables,
    rng=new_rng,
    ema_state=new_ema_state
  )
  # ----------------------------------------------------------------------------

  if dynamic_scale:
    # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
    # params should be restored (= skip this step).
    new_state = new_state.replace(
        opt_state=jax.tree_multimap(
            functools.partial(jnp.where, is_fin),
            new_state.opt_state,
            state.opt_state),
        params=jax.tree_multimap(
            functools.partial(jnp.where, is_fin),
            new_state.params,
            state.params))
    metrics['scale'] = dynamic_scale.scale

  return new_state, metrics


def eval_step(state, batch, ema_eval=False):
  variables = {'params': state.params, **state.variables}
  logits = state.apply_fn(variables, batch['image'], train=False, mutable=False)
  metrics = compute_metrics(logits, batch['label'], batch['label_one_hot'])
  metrics['test_acc1'] = metrics.pop('accuracy') * 100  # rename
  metrics['test_loss'] = metrics.pop('loss')  # rename

  if ema_eval:
    logits = state.apply_fn(state.ema_state.ema, batch['image'], train=False, mutable=False)
    metrics_ema = compute_metrics(logits, batch['label'], batch['label_one_hot'])
    metrics['test_acc1_ema'] = metrics_ema.pop('accuracy') * 100  # rename

  return metrics


def prepare_tf_data(xs):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


def create_input_iter(dataset_builder, batch_size, image_size, dtype, train,
                      cache, aug=None):
  ds = input_pipeline.create_split(
      dataset_builder, batch_size, image_size=image_size, dtype=dtype,
      train=train, cache=cache, aug=aug)

  if aug is not None and (aug.mix.mixup or aug.mix.cutmix):
    apply_mix = functools.partial(mix_util.apply_mix, cfg=aug.mix)
    ds = map(apply_mix, ds)

  # ------------------------------------------------
  # from IPython import embed; embed();
  # if (0 == 0): raise NotImplementedError
  # x = next(iter(ds))
  # ------------------------------------------------

  ds = map(prepare_tf_data, ds)
  it = jax_utils.prefetch_to_device(ds, 2)
  return it


class TrainState(train_state.TrainState):
  rng: Any
  variables: flax.core.FrozenDict[str, Any]
  # dynamic_scale: flax.optim.DynamicScale
  dynamic_scale: Any
  ema_tx: optax.GradientTransformation = struct.field(pytree_node=False)
  ema_state: optax.EmaState


def restore_checkpoint(state, workdir):
  return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
  if jax.process_index() == 0:
    # get train state from the first replica
    state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint(workdir, state, step, keep=3)


def profile_memory(workdir):
  jax.profiler.save_device_memory_profile("/tmp/memory.prof")
  if jax.process_index() == 0:
    os.system('gsutil cp /tmp/memory.prof {}'.format(workdir))


# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')


def sync_batch_stats(state):
  """Sync the batch statistics across replicas."""
  # Each device has its own version of the running average batch statistics and
  # we sync them before evaluation.
  if 'batch_stats' not in state.variables:
    return state
  else:
    new_variables, batch_stats = state.variables.pop('batch_stats')
    batch_stats = cross_replica_mean(batch_stats)
    return state.replace(variables=flax.core.FrozenDict({'batch_stats': batch_stats, **new_variables}))


def create_train_state(rng, config: ml_collections.ConfigDict,
                       model, image_size, learning_rate_fn):
  """Create initial training state."""
  dynamic_scale = None
  platform = jax.local_devices()[0].platform
  if config.half_precision and platform == 'gpu':
    dynamic_scale = optim.DynamicScale()
  else:
    dynamic_scale = None

  # split rng for init and for state
  rng_init, rng_state = jax.random.split(rng)

  variables = initialized(rng_init, image_size, model, config.init_backend)
  variables_states, params = variables.pop('params')

  # optional: rescale
  if config.rescale_init:
    rescales = opt_util.filter_parameters(params, opt_util.layer_rescale)
    params = jax.tree_util.tree_multimap(lambda x, y: x * y, rescales, params)

  stds = jax.tree_util.tree_map(lambda x: np.array(x).std(), params)
  # logging.info('std: {}'.format(stds))

  # optional: exclude some wd
  if config.exclude_wd:
    mask = jax.tree_util.tree_multimap(lambda x, y: bool(x and y), 
      opt_util.filter_parameters(params, opt_util.filter_bias_and_norm),
      opt_util.filter_parameters(params, opt_util.filter_cls_and_posembed)
    )
  else:
    mask = None
  # logging.info('Apply weight decay: {}'.format(mask))

  # tx = getattr(optax, config.opt_type)  # optax.adamw
  tx = getattr(adamw_util, config.opt_type)  # optax.adamw
  tx = tx(learning_rate=learning_rate_fn, **config.opt, mask=mask, mu_dtype=getattr(jnp, config.opt_mu_dtype))
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
      dynamic_scale=dynamic_scale,
      ema_tx=ema_tx,
      ema_state=ema_state)
  return state


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """

  writer = metric_writers.create_default_writer(
      logdir=workdir, just_logging=jax.process_index() != 0)

  rng = random.PRNGKey(0)

  image_size = 224

  if config.batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  local_batch_size = config.batch_size // jax.process_count()

  platform = jax.local_devices()[0].platform

  if config.half_precision:
    if platform == 'tpu':
      input_dtype = tf.bfloat16
    else:
      input_dtype = tf.float16
  else:
    input_dtype = tf.float32

  dataset_builder = tfds.builder(config.dataset)
  train_iter = create_input_iter(
      dataset_builder, local_batch_size, image_size, input_dtype, train=True,
      cache=config.cache, aug=config.aug)
  eval_iter = create_input_iter(
      dataset_builder, local_batch_size, image_size, input_dtype, train=False,
      cache=config.cache)

  steps_per_epoch = (
      dataset_builder.info.splits['train'].num_examples // config.batch_size
  )

  if config.num_train_steps == -1:
    num_steps = int(steps_per_epoch * config.num_epochs)
  else:
    num_steps = config.num_train_steps

  if config.steps_per_eval == -1:
    num_validation_examples = dataset_builder.info.splits[
        'validation'].num_examples
    steps_per_eval = num_validation_examples // config.batch_size
  else:
    steps_per_eval = config.steps_per_eval

  steps_per_checkpoint = int(steps_per_epoch * config.save_every_epochs)

  abs_learning_rate = config.learning_rate * config.batch_size / 256.

  # model_cls = getattr(models, config.model)
  model_cls = models_vit.VisionTransformer
  model = create_model(
      model_cls=model_cls, half_precision=config.half_precision, **config.model)

  learning_rate_fn = create_learning_rate_fn(
      config, abs_learning_rate, steps_per_epoch)

  state = create_train_state(rng, config, model, image_size, learning_rate_fn)

  if config.pretrain_dir == '':
    state = restore_checkpoint(state, workdir)
  else:
    logging.info('Loading from pre-training:')
    state = checkpoint_util.load_from_pretrain(state, config.pretrain_dir)

  # step_offset > 0 if restarting from checkpoint
  step_offset = int(state.step)

  # --------------------------------------------------------------------------------
  # up til now, state.params are for one device
  # image = jnp.ones([2, 224, 224, 3])
  # label = jnp.ones([2,], dtype=jnp.int32)
  # mutable_keys = [k for k in state.variables]
  # outcome = state.apply_fn(
  #     {'params': state.params,
  #      **state.variables,
  #     },
  #     rngs=dict(dropout=state.rng),
  #     inputs=image,
  #     mutable=mutable_keys,
  #     train=True)
  # logits, new_variables = outcome
  # num_params = np.sum([np.prod(p.shape) for p in jax.tree_leaves(state.opt_state[0].nu)])
  # num_params = np.sum([np.prod(p.shape) for p in jax.tree_leaves(state.params)])
  # num_params_mem = num_params * 4 / 1024 / 1024
  # --------------------------------------------------------------------------------  
  state = jax_utils.replicate(state)

  p_train_step = jax.pmap(
      functools.partial(train_step, learning_rate_fn=learning_rate_fn, config=config),
      axis_name='batch',
      donate_argnums=(0,) if config.donate else ()
      )
  p_eval_step = jax.pmap(
      functools.partial(eval_step, ema_eval=(config.ema and config.ema_eval)),
      axis_name='batch')

  train_metrics = []
  hooks = []
  # if jax.process_index() == 0:
  #   hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
  train_metrics_last_t = time.time()
  logging.info('Initial compilation, this might take some minutes...')

  for step, batch in zip(range(step_offset, num_steps), train_iter):
    state, metrics = p_train_step(state, batch)
    for h in hooks:
      h(step)
    if step == step_offset:
      logging.info('Initial compilation completed.')

    epoch_1000x = int(step * config.batch_size / 1281167 * 1000)  # normalize to IN1K epoch anyway

    if config.get('log_every_steps'):
      train_metrics.append(metrics)
      if (step + 1) % config.log_every_steps == 0:

        # if (step + 1) == config.log_every_steps and config.profile_memory:
        #   profile_memory(workdir)
        # Wait until computations are done before exiting
        jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
        train_metrics = common_utils.get_metrics(train_metrics)
        summary = {
            f'train_{k}': float(v)
            for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()
        }
        summary['steps_per_second'] = config.log_every_steps / (
            time.time() - train_metrics_last_t)

        # to make it consistent with PyTorch log
        summary['loss'] = summary['train_loss']  # add extra name
        summary['lr'] = summary.pop('train_learning_rate')  # rename
        summary['class_acc'] = summary.pop('train_accuracy')  # this is [0, 1]
        summary['step_tensorboard'] = epoch_1000x  # step for tensorboard

        writer.write_scalars(step + 1, summary)
        train_metrics = []
        train_metrics_last_t = time.time()

    if (step + 1) % steps_per_epoch == 0:
      epoch = step // steps_per_epoch
      eval_metrics = []

      # sync batch statistics across replicas
      state = sync_batch_stats(state)
      for _ in range(steps_per_eval):
        eval_batch = next(eval_iter)
        metrics = p_eval_step(state, eval_batch)
        eval_metrics.append(metrics)
      eval_metrics = common_utils.get_metrics(eval_metrics)
      summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
      values = [f"{k}: {v:.6f}" for k, v in sorted(summary.items())]
      logging.info('eval epoch: %d, %s', epoch, ', '.join(values))

      # to make it consistent with PyTorch log
      summary['step_tensorboard'] = epoch  # step for tensorboard (no need to minus 1)

      writer.write_scalars(step + 1, summary)
      writer.flush()

    if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
      state = sync_batch_stats(state)
      save_checkpoint(state, workdir)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  if config.profile_memory:
    profile_memory(workdir)

  return state
