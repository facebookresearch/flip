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
from flax.training import common_utils
from flax.training import train_state
import jax
import jax.numpy as jnp
from jax import random
import ml_collections
import optax
import tensorflow as tf

import models_vit

from utils import summary_util as summary_util  # must be after 'from clu import metric_writers'
from utils import checkpoint_util as ckp
from utils import torchloader_util
from utils import logging_util

from t5x.train_state_initializer import create_train_state
import t5x.partitioning
import t5x.rng
from t5x.model_info import log_model_info
import t5x.checkpoints

import jax.profiler

import numpy as np
import os
import random as _random

import torch
import torch.utils.data


def build_dataloaders(config, partitioner, rng_torch):

  data_layout = partitioner.get_data_layout(config.batch_size)
  shard_id = data_layout.shard_id
  num_shards = data_layout.num_shards

  # ----------------------------------------
  logging_util.verbose_on()
  logging_util.sync_and_delay()
  logging.info(data_layout)
  logging_util.verbose_off()
  # ----------------------------------------

  if config.batch_size % num_shards > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  local_batch_size = config.batch_size // num_shards

  dataset_val = torchloader_util.build_dataset(is_train=False, data_dir=config.torchload.data_dir, aug=config.aug)
  dataset_train = torchloader_util.build_dataset(is_train=True, data_dir=config.torchload.data_dir, aug=config.aug)

  sampler_train = torch.utils.data.DistributedSampler(
    dataset_train,
    num_replicas=num_shards, # jax.process_count(),
    rank=shard_id, # jax.process_index(),
    shuffle=True,
    seed=config.seed_pt,
  )
  sampler_val = torch.utils.data.DistributedSampler(
    dataset_val,
    num_replicas=num_shards, # jax.process_count(),
    rank=shard_id, # jax.process_index(),
    shuffle=False,
  )
  
  data_loader_train = torch.utils.data.DataLoader(
    dataset_train, sampler=sampler_train,
    batch_size=local_batch_size,
    num_workers=config.torchload.num_workers,
    pin_memory=True,
    drop_last=True,
    generator=rng_torch,
    worker_init_fn=functools.partial(seed_worker, shard_id=shard_id),
    persistent_workers=True,
    timeout=60.,
  )
  data_loader_val = torch.utils.data.DataLoader(
    dataset_val, sampler=sampler_val,
    batch_size=local_batch_size,
    num_workers=config.torchload.num_workers,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True,
    timeout=60.,
  )

  assert len(data_loader_train) == len(dataset_train) // config.batch_size
  return data_loader_train, data_loader_val, local_batch_size


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
  return metrics


def compute_eval_metrics(logits, labels, labels_one_hot):
  """kaiming: we do not average here (to support the reminder batch)
  """
  loss = optax.softmax_cross_entropy(logits=logits, labels=labels_one_hot)
  accuracy = (jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
      'label': labels
  }
  metrics = jax.tree_map(lambda x: jnp.reshape(x, [-1,]), metrics)
  return metrics


def train_step(state, batch, model, rng):
  """Perform a single training step."""
  dropout_rng = jax.random.fold_in(rng, state.step)

  def loss_fn(params):
    """loss function used for training."""
    mutable = [k for k in state.flax_mutables]
    outcome = model.apply(
        {'params': params, **state.flax_mutables},
        inputs=batch['image'],
        mutable=mutable,
        rngs=dict(dropout=dropout_rng),
        train=True)
    logits, new_mutables = outcome

    loss = cross_entropy_loss(logits, batch['label_one_hot'])
    return loss, (new_mutables, logits)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  aux, grads = grad_fn(state.params)

  new_mutables, logits = aux[1]
  metrics = compute_metrics(logits, batch['label'], batch['label_one_hot'])

  # only for metric logging
  lr = state._optimizer.optimizer_def.metric_learning_rate_fn(state.step)
  metrics['learning_rate'] = lr

  new_state = state.apply_gradient(
    grads,
    learning_rate=None,  # TODO: not used in adamw
    flax_mutables=new_mutables)
  return new_state, metrics


def eval_step(state, batch, model):
  variables = {'params': state.params, **state.flax_mutables}
  logits = model.apply(variables, batch['image'], train=False, mutable=False)
  metrics = compute_eval_metrics(logits, batch['label'], batch['label_one_hot'])
  metrics['test_acc1'] = metrics.pop('accuracy') * 100  # rename
  metrics['perf/test_acc1'] = metrics['test_acc1']  # for comparing with pytorch
  metrics['test_loss'] = metrics.pop('loss')  # rename

  return metrics


def parse_batch(batch, local_batch_size, mixup_fn=None):
  images, labels, labels_one_hot = batch
  if mixup_fn is not None:
    assert images.shape[1] == 3  # nchw
    images, labels_one_hot = mixup_fn(images, labels)
  images = images.permute([0, 2, 3, 1])  # nchw -> nhwc
  batch = {'image': images, 'label': labels, 'label_one_hot': labels_one_hot}
  batch = prepare_pt_data(batch, local_batch_size)  # to (local_devices, device_batch_size, height, width, 3)
  return batch


def prepare_pt_data(xs, batch_size):
  """Convert a input batch from PyTorch Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x.numpy()  # pylint: disable=protected-access

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
  rng_torch = set_seed_torch(config.seed_pt)
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
  data_loader_train, data_loader_val, local_batch_size = build_dataloaders(config, partitioner, rng_torch)

  mixup_fn = torchloader_util.get_mixup_fn(config.aug)

  steps_per_epoch = len(data_loader_train)
  
  # ------------------------------------
  # Create model
  # ------------------------------------
  model = models_vit.VisionTransformer(**config.model)
  
  p_init_fn, state_axes, state_shape = create_train_state(config, model, image_size, steps_per_epoch, partitioner)
  rng_init, rng = jax.random.split(rng)

  log_model_info(None, state_shape, partitioner)
  # profile_memory(workdir)

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
    # When fine-tuning, we run initialization anyway
    logging.info('Initializing train_state...')
    state = p_init_fn(rng_init)
    logging.info('Initializing train_state done.')

    state = ckp.restore_from_pretrain(state, config, partitioner, state_axes)
  else:
    logging.info('Initializing train_state...')
    state = p_init_fn(rng_init)
    logging.info('Initializing train_state done.')
    # stds = jax.tree_util.tree_map(lambda x: np.array(x).std(), state.params)
    # logging.info('std: {}'.format(stds))

  # debug
  # checkpointer.save(state)
  # state = checkpointer.restore(path=checkpointer.checkpoints_dir + '/checkpoint_0')

  # step_offset > 0 if restarting from checkpoint
  step_offset = int(state.step)
  logging.info('step_offset: {}'.format(step_offset))

  # to create partitioned train_step
  train_step_fn = functools.partial(train_step, model=model, rng=rng)  # (state, batch, rng) -> (state, metrics)
  partitioned_train_step = partitioner.partition(
        train_step_fn,
        in_axis_resources=(state_axes, partitioner.data_partition_spec),
        out_axis_resources=(state_axes, None),
        donate_argnums=(0,))

  eval_step_fn = functools.partial(eval_step, model=model)  # (state, batch) -> metrics
  partitioned_eval_step = partitioner.partition(
        eval_step_fn,
        in_axis_resources=(state_axes, partitioner.data_partition_spec),
        out_axis_resources=None)

  # ------------------------------------------
  # debug
  # batch = next(iter(data_loader_train))
  # batch = parse_batch(batch, local_batch_size, mixup_fn)
  # metrics = partitioned_eval_step(state, batch)
  # ------------------------------------------

  train_metrics = []

  logging.info('Work dir: {}'.format(workdir))
  train_metrics_last_t = time.time()
  logging.info('Initial compilation, this might take some minutes...')

  if config.eval_only:
    # run eval only and return
    logging.info('Evaluating...')
    run_eval(state, partitioned_eval_step, data_loader_val, local_batch_size, epoch=-1)
    return

  epoch_offset = (step_offset + 1) // steps_per_epoch
  step = epoch_offset * steps_per_epoch

  # assert step == int(jnp.reshape(state.step, (-1,))[0])  # sanity when loading
  data_layout = partitioner.get_data_layout(config.batch_size)
  shard_id = data_layout.shard_id

  best_acc = 0.
  for epoch in range(epoch_offset, int(config.num_epochs)):
    data_loader_train.sampler.set_epoch(epoch)  # reset random seed
    
    # ------------------------------------------------------------
    # train one epoch
    # ------------------------------------------------------------
    logging_util.sync_and_delay(10)
    for i, batch in enumerate(data_loader_train):
      batch = parse_batch(batch, local_batch_size, mixup_fn)
      state, metrics = partitioned_train_step(state, batch)

      if epoch == epoch_offset and i == 0 and partitioner._num_partitions > 8:
        print_sanity_check(batch, shard_id)

      epoch_1000x = int(step * config.batch_size / 1281167 * 1000)  # normalize to IN1K epoch anyway

      if epoch == epoch_offset and i == 0:
        logging.info('Initial compilation completed.')
        start_time = time.time()  # log the time after compilation

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
          summary['class_acc'] = summary.pop('train_accuracy')  # this is [0, 1]
          summary['step_tensorboard'] = epoch_1000x  # step for tensorboard

          writer.write_scalars(step + 1, summary)
          train_metrics = []
          train_metrics_last_t = time.time()

      step += 1  
    # ------------------------------------------------------------
    # finished one epoch: eval
    # ------------------------------------------------------------
    if True:
      summary = run_eval(state, partitioned_eval_step, data_loader_val, local_batch_size, epoch)
      best_acc = max(best_acc, summary['test_acc1'])

      # to make it consistent with PyTorch log
      summary['step_tensorboard'] = epoch  # step for tensorboard (no need to minus 1)

      writer.write_scalars(step + 1, summary)
      writer.flush()

    # ------------------------------------------------------------
    # finished one epoch: eval
    # ------------------------------------------------------------
    if (epoch + 1) % config.save_every_epochs == 0 or epoch + 1 == int(config.num_epochs):
      logging.info('Saving checkpoint: {}'.format(workdir))
      checkpointer.save(state)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
  total_time = time.time() - start_time
  total_time_str = str(datetime.timedelta(seconds=int(total_time)))
  logging.info('Last accuracy: {}, best accuracy: {}'.format(summary['test_acc1'], best_acc))
  logging.info('Elapsed time: {}'.format(total_time_str))

  if config.profile_memory:
    profile_memory(workdir)
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  return state


def run_eval(state, partitioned_eval_step, data_loader_val, local_batch_size, epoch):
  eval_metrics = []
  tic = time.time()
  for _, batch in enumerate(data_loader_val):
    batch = parse_batch(batch, local_batch_size, mixup_fn=None)
    metrics = partitioned_eval_step(state, batch)
    eval_metrics.append(metrics)
    # logging.info('{} / {}'.format(_, len(data_loader_val)))

  # eval_metrics = jax.tree_map(lambda x: x, eval_metrics)
  eval_metrics = jax.device_get(eval_metrics)
  eval_metrics = jax.tree_map(lambda *args: np.concatenate(args), *eval_metrics)

  valid = np.where(eval_metrics['label'] >= 0)  # remove padded patch
  eval_metrics.pop('label')
  eval_metrics = jax.tree_util.tree_map(lambda x: x[valid], eval_metrics)

  toc = time.time() - tic
  logging.info('Eval time: {}, {} steps, {} samples'.format(
    str(datetime.timedelta(seconds=int(toc))),
    len(data_loader_val),
    len(eval_metrics['test_acc1'])))

  summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
  values = [f"{k}: {v:.6f}" for k, v in sorted(summary.items())]
  logging.info('eval epoch: %d, %s', epoch, ', '.join(values))
  return summary