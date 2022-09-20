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

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

import functools
import time, datetime
from typing import Any

from absl import logging
from clu import metric_writers
import flax
from flax import jax_utils
from flax.training import common_utils
from flax.training import train_state
import jax
import jax.numpy as jnp
from jax import random
from jax.interpreters.sharded_jit import PartitionSpec
import ml_collections
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

import input_pipeline
import input_pipeline_laion
import models_mae

from utils import summary_util as summary_util  # must be after 'from clu import metric_writers'
from utils import checkpoint_util as ckp
from utils import torchloader_util
from utils import logging_util
from utils.torchloader_util import MEAN_RGB, STDDEV_RGB

from t5x.train_state_initializer import create_train_state
import t5x.partitioning
import t5x.rng
import t5x.model_info
import t5x.checkpoints

import jax.profiler

import math
import numpy as np
import os
import random as _random

import torch
import torch.utils.data


def create_imagenet_input_iter(local_batch_size, data_layout, image_size, dtype, train, cache, seed=0, aug=None,):
  dataset_builder = tfds.builder('imagenet2012:5.*.*')
  ds = input_pipeline.create_split(
      dataset_builder, local_batch_size, data_layout, image_size=image_size, dtype=dtype,
      train=train, cache=cache, seed=seed, aug=aug,)

  ds = map(functools.partial(prepare_tf_data, batch_size=local_batch_size), ds)
  return ds


def create_laion_input_iter(local_batch_size, data_layout, image_size, dtype, train,
                      cache, seed=0, cfg=None, from_tags=None):
  ds = input_pipeline_laion.create_split(
      local_batch_size, data_layout, image_size=image_size, dtype=dtype,
      train=train, cache=cache, seed=seed, cfg=cfg, from_tags=from_tags)

  # ------------------------------------------------
  # x = next(iter(ds))
  # ------------------------------------------------

  ds = map(functools.partial(prepare_tf_data, batch_size=local_batch_size), ds)
  return ds


def build_dataloaders(config, partitioner):

  batch_size = config.batch_size

  data_layout = partitioner.get_data_layout(batch_size)
  shard_id = data_layout.shard_id
  num_shards = data_layout.num_shards

  if batch_size % num_shards > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  local_batch_size = batch_size // num_shards

  # ----------------------------------------
  logging_util.verbose_on()
  logging_util.sync_and_delay()
  logging.info("shard_id: {}".format(shard_id))
  logging_util.verbose_off()
  # ----------------------------------------

  image_size = config.image_size
  input_dtype = tf.float32

  # ImageNet tags
  from vocab.class_names import CLIP_IMAGENET_CLASS_NAMES
  # CLIP_IMAGENET_TEMPLATES_FULL, CLIP_IMAGENET_TEMPLATES_SHORT, CLIP_IMAGENET_TEMPLATES_NONE
  from vocab.class_names import CLIP_IMAGENET_TEMPLATES_SHORT as templates

  tags = []
  for c in CLIP_IMAGENET_CLASS_NAMES:
    for t in templates:
      tags.append(t(c))

  data_loader_tags = create_laion_input_iter(
      8,  # local_batch_size=8
      data_layout,
      image_size,
      input_dtype,
      train=False,
      cache=False, # config.cache, 
      seed=config.seed_tf,
      cfg=config,
      from_tags=tags)

  data_loader_train = create_laion_input_iter(
      local_batch_size,
      data_layout,
      image_size,
      input_dtype,
      train=True,
      cache=False, # config.cache, 
      seed=config.seed_tf,
      cfg=config)

  # val set is imagenet
  data_loader_val = create_imagenet_input_iter(
      local_batch_size,
      data_layout,
      image_size,
      input_dtype,
      train=False,
      cache=config.cache, 
      seed=config.seed_tf,
      aug=config.aug)
  # data_loader_val = None

  return data_loader_train, data_loader_val, data_loader_tags


def print_sanity_check(batch, shard_id):
  """A sanity check when model partitions > 8 and data must be shared across nodes
  """
  logging_util.sync_and_delay(delay=shard_id * 0.5)
  logging_util.verbose_on()
  str = '{}'.format(batch['label'])
  str = (str + ' ' * 60)[:60] + '...'
  logging.info('shard: {}, label: {}'.format(shard_id, str))

  logging_util.sync_and_delay(delay=shard_id * 0.5)
  str = '{}'.format(np.array(batch['image'][:, 0, 0, 0]))
  str = (str + ' ' * 60)[:60] + '...'
  logging.info('shard: {}, image: {}'.format(shard_id, str))
  logging_util.verbose_off()
  return


def train_step(state, batch, model, rng):
  """Perform a single training step."""
  dropout_rng = jax.random.fold_in(rng, state.step)

  def loss_fn(params):
    """loss function used for training."""
    mutable = [k for k in state.flax_mutables]
    outcome = model.apply(
        {'params': params, **state.flax_mutables},
        inputs=batch,
        mutable=mutable,
        rngs=dict(dropout=dropout_rng),
        train=True)
    (loss, _, artifacts), new_mutables = outcome
    return loss, (new_mutables, artifacts)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  aux, grads = grad_fn(state.params)

  new_mutables, artifacts = aux[1]

  # metrics = {'loss': loss}
  metrics = {**artifacts}

  # only for metric logging
  lr = state._optimizer.optimizer_def.metric_learning_rate_fn(state.step)
  metrics['learning_rate'] = lr

  new_state = state.apply_gradient(
    grads,
    learning_rate=None,  # TODO: not used in adamw
    flax_mutables=new_mutables)
  return new_state, metrics


def eval_step(state, batch, encoded_tags, model, rng):
  variables = {'params': state.params, **state.flax_mutables}

  dropout_rng = jax.random.fold_in(rng, state.step)

  outcome = model.apply(variables, batch, train=False, mutable=False, rngs=dict(dropout=dropout_rng), encode_txt=False)
  loss, _, artifacts = outcome
  z_img = artifacts['z_img']

  labels = batch['label']

  z_txt = encoded_tags
  logits = jnp.einsum('nc,mc->nm', z_img, z_txt)

  # --------
  # dev: for maxout multiple templates
  # logits = logits.reshape([logits.shape[0], 1000, -1])
  # logits = jnp.max(logits, axis=-1)
  # --------

  pred_labels = jnp.argmax(logits, -1)
  accuracy = jnp.float32(pred_labels == labels)
  metrics = {'test_acc1': accuracy, 'label': labels}
  metrics = jax.tree_map(lambda x: jnp.reshape(x, [-1,]), metrics)
  return metrics


def eval_tags_step(state, batch, model, rng):
  variables = {'params': state.params, **state.flax_mutables}

  dropout_rng = jax.random.fold_in(rng, state.step)

  outcome = model.apply(variables, batch, train=False, mutable=False, rngs=dict(dropout=dropout_rng), encode_img=False)
  loss, _, artifacts = outcome
  z_txt = artifacts['z_txt']

  # metrics = {'test_loss': loss, 'imgs_vis': imgs_vis}
  return z_txt


def prepare_tf_data(xs, batch_size):
  """Convert a input batch from PyTorch Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    if x.shape[0] != batch_size:
      pads = -np.ones((batch_size - x.shape[0],) + x.shape[1:], dtype=x.dtype)
      x = np.concatenate([x, pads], axis=0)

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    # return x.reshape((local_device_count, -1) + x.shape[1:])
    return x.reshape((-1,) + x.shape[1:])  # do not reshape into (local_devices, -1, ...)

  return jax.tree_map(_prepare, xs)


def profile_memory(workdir):
  jax.profiler.save_device_memory_profile("/tmp/memory.prof")
  if jax.process_index() == 0:
    logging.info('Saving memory.prof...')
    os.system('cd ~; gsutil cp /tmp/memory.prof {}'.format(workdir))
    logging.info('Saved memory.prof.')


def seed_worker(worker_id, shard_id):
    # worker_seed = torch.initial_seed() % 2**32 + shard_id
    worker_seed = worker_id + shard_id * 10000
    np.random.seed(worker_seed)
    _random.seed(worker_seed)

    # logging_util.verbose_on()
    # logging.info('worker_id: {}, shard_id: {}, worker_seed: {}'.format(worker_id, shard_id, worker_seed))
    # logging_util.verbose_off()


def set_seed_torch(seed):
  rng_torch = torch.Generator()
  rng_torch.manual_seed(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)
  _random.seed(seed)
  return rng_torch


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> train_state.TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """
  # ------------------------------------
  # Set random seeds
  # ------------------------------------
  # rng_torch = set_seed_torch(config.seed_pt)
  tf.random.set_seed(config.seed_tf + jax.process_index())

  t5x.rng.set_hardware_rng_ops()
  rng = random.PRNGKey(config.seed_jax)
  # ------------------------------------

  writer = metric_writers.create_default_writer(
      logdir=workdir, just_logging=jax.process_index() != 0)

  image_size = 224  # TODO: move to config and model

  # ------------------------------------
  # Create partitioner
  # ------------------------------------
  partitioner = t5x.partitioning.PjitPartitioner(**config.partitioning)
  partitioner._logical_axis_rules += (('_null0', None),)
  partitioner._logical_axis_rules += (('_null1', None),)
  partitioner._logical_axis_rules += (('_null2', None),)
  partitioner._logical_axis_rules += (('classes', None),)

  # ------------------------------------
  # Create data loader
  # ------------------------------------
  data_loader_train, data_loader_val, data_loader_tags = build_dataloaders(config, partitioner)  # we do not use data_loader_val
  batched_tags = [d for d in data_loader_tags]  # 1000x80 or 1000x7

  steps_per_epoch = config.samples_per_epoch // config.batch_size  # for lr schedule
  
  # ------------------------------------
  # Create model
  # ------------------------------------
  model = models_mae.ImageTextLearner(config=config.model)
  
  p_init_fn, state_axes, state_shape = create_train_state(
    config, model, steps_per_epoch, partitioner, init_batch=next(data_loader_train))
  rng_init, rng = jax.random.split(rng)

  t5x.model_info.log_model_info(None, state_shape, partitioner)

  # ------------------------------------
  # Create checkpointer
  # ------------------------------------
  checkpointer = t5x.checkpoints.Checkpointer(
    train_state=state_shape,
    partitioner=partitioner,
    checkpoints_dir=workdir,
    keep=None,  # TODO: move to config
  )
  
  if config.resume_dir != '':
    state = ckp.restore_checkpoint(checkpointer, path=config.resume_dir)
  elif config.pretrain_dir != '':
    raise NotImplementedError
  else:
    logging.info('Initializing train_state...')
    state = p_init_fn(rng_init)
    logging.info('Initializing train_state done.')
    # stds = jax.tree_util.tree_map(lambda x: (x.shape, np.array(x).std()), state.params)
    # logging.info('std: {}'.format(stds))

  t5x.model_info.log_state_info(state)

  # ------------------------------------------
  # for debugging with real tensors
  # batch = next(iter(data_loader_train))
  # mutable = [k for k in state.flax_mutables]
  # outcome = model.apply(
  #     {'params': state.params, **state.flax_mutables},
  #     inputs=batch,
  #     mutable=mutable,
  #     rngs=dict(dropout=rng),
  #     train=True)
  # # use the following to add checkpoints
  # import jaxlib
  # if isinstance(x, jnp.DeviceArray):
  #   pass
  # ------------------------------------------

  # --------------------------------------------------------
  # logging.info('Saving debug checkpoint: {}'.format(workdir))
  # checkpointer.save(state)
  # --------------------------------------------------------

  # step_offset > 0 if restarting from checkpoint
  step_offset = int(state.step)
  logging.info('step_offset: {}'.format(step_offset))

  # ------------------------------------------
  # build eval_tags_step
  eval_step_fn = functools.partial(eval_tags_step, model=model, rng=rng)  # (state, batch) -> metrics
  eval_axes = PartitionSpec('data', None,)
  partitioned_eval_tags_step = partitioner.partition(
        eval_step_fn,
        in_axis_resources=(state_axes, partitioner.data_partition_spec),
        out_axis_resources=eval_axes)

  # ------------------------------------------
  # batch = next(iter(data_loader_tags))
  # logging.info('To run eval_tags_step:')
  # z_txt = eval_tags_step(state, batch, model=model, rng=rng)
  # z_txt = partitioned_eval_tags_step(state, batch)
  # ------------------------------------------

  # ------------------------------------------
  # to create partitioned train_step
  train_step_fn = functools.partial(train_step, model=model, rng=rng)  # (state, batch) -> (state, metrics)
  partitioned_train_step = partitioner.partition(
        train_step_fn,
        in_axis_resources=(state_axes, partitioner.data_partition_spec),
        out_axis_resources=(state_axes, None),
        donate_argnums=(0,))

  eval_step_fn = functools.partial(eval_step, model=model, rng=rng)  # (state, batch) -> metrics
  eval_axes = None
  partitioned_eval_step = partitioner.partition(
        eval_step_fn,
        in_axis_resources=(state_axes, partitioner.data_partition_spec, None),
        out_axis_resources=eval_axes)
  # ------------------------------------------

  # ------------------------------------------
  # debug
  # encoded_tags = compute_encoded_tags(state, batched_tags, partitioned_eval_tags_step)
  # batch = next(iter(data_loader_val))
  # logging.info('To run eval_step:')
  # outcome = eval_step(state, batch, encoded_tags, model=model, rng=rng)
  # outcome = partitioned_eval_step(state, batch, encoded_tags)
  # logging.info(jax.tree_map(lambda x: x.shape, outcome))
  # ------------------------------------------


  # ------------------------------------------
  if config.eval_only:
    logging.info('Eval only...')
    summary = run_eval(
      state,
      batched_tags,
      partitioned_eval_tags_step,
      data_loader_val,
      partitioned_eval_step,
      config,)
    values = [f"{k}: {v:.6f}" for k, v in sorted(summary.items())]
    logging.info('eval: %s', ', '.join(values))
    return
  # ------------------------------------------

  train_metrics = []

  logging.info('Work dir: {}'.format(workdir))
  train_metrics_last_t = time.time()
  logging.info('Initial compilation, this might take some minutes...')

  epoch_offset = (step_offset + 1) // steps_per_epoch
  step = epoch_offset * steps_per_epoch

  # assert step == int(jnp.reshape(state.step, (-1,))[0])  # sanity when loading
  data_layout = partitioner.get_data_layout(config.batch_size)
  shard_id = data_layout.shard_id

  for epoch in range(epoch_offset, int(config.num_epochs)):
    # data_loader_train.sampler.set_epoch(epoch)  # reset random seed
    
    # ------------------------------------------------------------
    # train one epoch (one "virtual" epoch)
    # ------------------------------------------------------------
    for i in range(steps_per_epoch):
      batch = next(data_loader_train)
      state, metrics = partitioned_train_step(state, batch)

      if epoch == epoch_offset and i == 0 and partitioner._num_partitions > 8:
        print_sanity_check(batch, shard_id)

      epoch_1000x = int(step * config.batch_size / 1281167 * 1000)  # normalize to IN1K epoch anyway

      if epoch == epoch_offset and i == 0:
        logging.info('Initial compilation completed.')
        start_time = time.time()  # log the time after compilation

      # if epoch == epoch_offset and i == 0:
      #   jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
      #   logging.info('Saving init debug checkpoint: {}'.format(workdir))
      #   checkpointer.save(state)

      if config.get('log_every_steps'):
        train_metrics.append(metrics)
        if (step + 1) % config.log_every_steps == 0:
          # Wait until computations are done before exiting
          jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
          train_metrics = common_utils.get_metrics(jax.tree_map(lambda x: jnp.reshape(x, (-1,)), train_metrics))
          summary = {
              f'train_{k}': float(v)
              for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()
          }
          summary['steps_per_second'] = config.log_every_steps / (
              time.time() - train_metrics_last_t)

          # to make it consistent with PyTorch log
          summary['loss'] = summary['train_loss']  # add extra name
          summary['lr'] = summary.pop('train_learning_rate')  # rename
          summary['step_tensorboard'] = epoch_1000x  # step for tensorboard

          writer.write_scalars(step + 1, summary)
          train_metrics = []
          train_metrics_last_t = time.time()

      step += 1  

    # ------------------------------------------------------------
    # finished one epoch: eval
    # ------------------------------------------------------------
    if (epoch + 1) % config.vis_every_epochs == 0 or epoch == epoch_offset:
      # --------------------------------------------------------------------
      summary = run_eval(
        state,
        batched_tags,
        partitioned_eval_tags_step,
        data_loader_val,
        partitioned_eval_step,
        config,)
      values = [f"{k}: {v:.6f}" for k, v in sorted(summary.items())]
      logging.info('eval epoch: %d, %s', epoch, ', '.join(values))

      # to make it consistent with PyTorch log
      summary['step_tensorboard'] = epoch  # step for tensorboard (no need to minus 1)
      writer.write_scalars(step + 1, summary)
      writer.flush()

      # --------------------------------------------------------------------

      # eval_batch = batch  # we visualize the same bach for simplicty
      # metrics = partitioned_eval_step(state, eval_batch)

      # imgs_vis = metrics.pop('imgs_vis')
      # if imgs_vis is not None:
      #   imgs_vis = imgs_vis * jnp.asarray(STDDEV_RGB) + jnp.asarray(MEAN_RGB)
      #   imgs_vis = jnp.uint8(jnp.clip(imgs_vis, 0, 255.))
      #   writer.write_images(step=epoch_1000x, images=dict(imgs_vis=imgs_vis))

      # summary = jax.tree_map(lambda x: x.mean(), metrics)
      # values = [f"{k}: {v:.6f}" for k, v in sorted(summary.items())]
      # logging.info('eval epoch: %d, %s', epoch, ', '.join(values))

    # ------------------------------------------------------------
    # finished one epoch: save
    # ------------------------------------------------------------
    if (epoch + 1) % config.save_every_epochs == 0 or epoch + 1 == int(config.num_epochs) or epoch == epoch_offset:
      logging.info('Saving checkpoint: {}'.format(workdir))
      checkpointer.save(state)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
  total_time = time.time() - start_time
  total_time_str = str(datetime.timedelta(seconds=int(total_time)))
  logging.info('Elapsed time: {}'.format(total_time_str))

  if config.profile_memory:
    profile_memory(workdir)
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  return state


def compute_encoded_tags(
  state,
  batched_tags,
  partitioned_eval_tags_step,
):
  # Encoding tags: no data-parallism across nodes
  logging.info('Encoding tags...')
  encoded_tags = []
  for i, tags_batch in enumerate(batched_tags):
    z_txt = partitioned_eval_tags_step(state, tags_batch)
    encoded_tags.append(z_txt)
    if i % 100 == 0:
      logging.info('{} / {}'.format(i, len(batched_tags)))
  encoded_tags = jnp.concatenate(encoded_tags, axis=0)  # type: DeviceArray

  # ----------------
  # average multiple templates
  encoded_tags = encoded_tags.reshape([1000, -1, encoded_tags.shape[-1]])  # [1000, 7, 512]
  encoded_tags = encoded_tags.mean(axis=1)
  encoded_tags /= jnp.linalg.norm(encoded_tags, axis=-1, keepdims=True) + 1e-8
  assert encoded_tags.shape[0] == 1000
  # ----------------

  logging.info('Encoding tags done: {}'.format(encoded_tags.shape))
  return encoded_tags


def run_eval(
  state,
  batched_tags,
  partitioned_eval_tags_step,
  data_loader_val,
  partitioned_eval_step,
  config,
):
  tic = time.time()
  encoded_tags = compute_encoded_tags(state, batched_tags, partitioned_eval_tags_step)

  steps_per_eval = math.ceil(50000 / config.batch_size)
  eval_metrics = []
  for i in range(steps_per_eval):
    eval_batch = next(data_loader_val)
    metrics = partitioned_eval_step(state, eval_batch, encoded_tags)
    eval_metrics.append(metrics)
    if config.eval_only and i % 10 == 0:
      logging.info('{} / {}, shape: {}'.format(i, steps_per_eval, eval_batch['image'].shape))

  eval_metrics = jax.device_get(eval_metrics)
  eval_metrics = jax.tree_map(lambda *args: np.concatenate(args), *eval_metrics)

  valid = np.where(eval_metrics['label'] >= 0)  # remove padded patch
  eval_metrics.pop('label')
  eval_metrics = jax.tree_util.tree_map(lambda x: x[valid], eval_metrics)

  toc = time.time() - tic
  logging.info('Eval time: {}, {} steps, {} samples'.format(
    str(datetime.timedelta(seconds=int(toc))),
    steps_per_eval,
    len(eval_metrics['test_acc1'])))

  summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
  return summary
