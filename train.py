# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# References:
# https://github.com/google/flax/tree/main/examples/imagenet


import datetime
import time
import os
import numpy as np
import math
import jax.profiler

import tensorflow_datasets as tfds
import tensorflow as tf
import ml_collections
from jax import random
import jax.numpy as jnp
import jax
from flax.training import train_state
from flax.training import common_utils
from clu import metric_writers
from absl import logging
import functools
import warnings

import t5x.checkpoints
import t5x.model_info
import t5x.rng
import t5x.partitioning
from t5x.train_state_initializer import create_train_state


import models_flip
import input_pipeline_laion
import input_pipeline_imagenet
from utils import logging_util
from utils import checkpoint_util as ckp

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


try:
    from jax.interpreters.sharded_jit import PartitionSpec
except ImportError:
    from jax.interpreters.pxla import PartitionSpec


def prepare_tf_data(xs, batch_size):
    """Convert a input batch from tf Tensors to numpy arrays."""

    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        x = x._numpy()  # pylint: disable=protected-access

        if x.shape[0] != batch_size:
            pads = -np.ones((batch_size - x.shape[0],) + x.shape[1:], dtype=x.dtype)
            x = np.concatenate([x, pads], axis=0)

        # do not reshape into (local_devices, -1, ...)
        return x.reshape((-1,) + x.shape[1:])

    return jax.tree_map(_prepare, xs)


def build_dataloaders(config, partitioner):
    batch_size = config.batch_size

    data_layout = partitioner.get_data_layout(batch_size)
    shard_id = data_layout.shard_id
    num_shards = data_layout.num_shards

    if batch_size % num_shards > 0:
        raise ValueError("Batch size must be divisible by the number of devices")
    local_batch_size = batch_size // num_shards

    # ----------------------------------------
    logging_util.verbose_on()
    logging_util.sync_and_delay()
    logging.info("shard_id: {}".format(shard_id))
    logging_util.verbose_off()
    # ----------------------------------------

    image_size = config.image_size
    input_dtype = tf.float32

    # training set is LAION.
    data_loader_train = input_pipeline_laion.create_split(
        config.laion_path,
        local_batch_size,
        data_layout,
        image_size=image_size,
        train=True,
        cache=False,
        seed=config.seed_tf,
        cfg=config,
    )
    data_loader_train = map(
        functools.partial(prepare_tf_data, batch_size=local_batch_size),
        data_loader_train,
    )

    # val set is imagenet
    data_loader_val = input_pipeline_imagenet.create_split(
        tfds.builder(config.eval_dataset),
        local_batch_size,
        data_layout,
        image_size=image_size,
        dtype=input_dtype,
        train=False,
        cache=config.cache,
        seed=config.seed_tf,
        aug=config.aug,
    )
    data_loader_val = map(
        functools.partial(prepare_tf_data, batch_size=local_batch_size),
        data_loader_val,
    )

    # ImageNet tags
    from vocab.class_names import CLIP_IMAGENET_CLASS_NAMES

    imagenet_templates = config.get("imagenet_templates", "short")
    if imagenet_templates == "short":
        from vocab.class_names import CLIP_IMAGENET_TEMPLATES_SHORT as templates

        tag_batch_size = 8
    elif imagenet_templates == "long":
        from vocab.class_names import CLIP_IMAGENET_TEMPLATES_FULL as templates

        tag_batch_size = 64
    else:
        raise NotImplementedError

    tags = []
    for c in CLIP_IMAGENET_CLASS_NAMES:
        for t in templates:
            tags.append(t(c))

    print(f"length of templates: {len(templates)}")
    data_loader_tags = input_pipeline_laion.create_tags_split(
        tags,
        tag_batch_size,
        image_size=None,
        train=False,
        cache=False,
        seed=config.seed_tf,
        cfg=config,
    )
    data_loader_tags = map(
        functools.partial(prepare_tf_data, batch_size=tag_batch_size),
        data_loader_tags,
    )

    return data_loader_train, data_loader_val, data_loader_tags


def print_sanity_check(batch, shard_id):
    """A sanity check when model partitions > 8 and data must be shared across nodes"""
    logging_util.sync_and_delay(delay=shard_id * 0.5)
    logging_util.verbose_on()
    str = "{}".format(batch["label"])
    str = (str + " " * 60)[:60] + "..."
    logging.info("shard: {}, label: {}".format(shard_id, str))

    logging_util.sync_and_delay(delay=shard_id * 0.5)
    str = "{}".format(np.array(batch["image"][:, 0, 0, 0]))
    str = (str + " " * 60)[:60] + "..."
    logging.info("shard: {}, image: {}".format(shard_id, str))
    logging_util.verbose_off()
    return


def train_step(state, batch, model, rng):
    """Perform a single training step."""
    dropout_rng = jax.random.fold_in(rng, state.step)

    def loss_fn(params):
        """loss function used for training."""
        mutable = [k for k in state.flax_mutables]
        outcome = model.apply(
            {"params": params, **state.flax_mutables},
            inputs=batch,
            mutable=mutable,
            rngs=dict(dropout=dropout_rng),
            train=True,
        )
        (loss, artifacts), new_mutables = outcome
        return loss, (new_mutables, artifacts)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)

    new_mutables, artifacts = aux[1]

    metrics = {**artifacts}

    # only for metric logging
    lr = state._optimizer.optimizer_def.metric_learning_rate_fn(state.step)
    metrics["learning_rate"] = lr

    new_state = state.apply_gradient(
        grads, learning_rate=None, flax_mutables=new_mutables  # TODO: not used in adamw
    )

    return new_state, metrics


def eval_step(state, batch, encoded_tags, model, rng):
    variables = {"params": state.params, **state.flax_mutables}

    dropout_rng = jax.random.fold_in(rng, state.step)

    outcome = model.apply(
        variables,
        batch,
        train=False,
        mutable=False,
        rngs=dict(dropout=dropout_rng),
        encode_txt=False,
    )
    _, artifacts = outcome
    z_img = artifacts["z_img"]

    labels = batch["label"]

    z_txt = encoded_tags
    logits = jnp.einsum("nc,mc->nm", z_img, z_txt)

    pred_labels = jnp.argmax(logits, -1)
    accuracy = jnp.float32(pred_labels == labels)
    metrics = {"test_acc1": accuracy, "label": labels}
    metrics = jax.tree_map(
        lambda x: jnp.reshape(
            x,
            [
                -1,
            ],
        ),
        metrics,
    )
    return metrics


def eval_tags_step(state, batch, model, rng):
    variables = {"params": state.params, **state.flax_mutables}

    dropout_rng = jax.random.fold_in(rng, state.step)

    outcome = model.apply(
        variables,
        batch,
        train=False,
        mutable=False,
        rngs=dict(dropout=dropout_rng),
        encode_img=False,
    )
    _, artifacts = outcome
    z_txt = artifacts["z_txt"]

    return z_txt


def train_and_evaluate(
    config: ml_collections.ConfigDict, workdir: str
) -> train_state.TrainState:
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
    tf.random.set_seed(config.seed_tf + jax.process_index())
    t5x.rng.set_hardware_rng_ops()
    rng = random.PRNGKey(config.seed_jax)
    # ------------------------------------

    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0
    )

    # ------------------------------------
    # Create partitioner
    # ------------------------------------
    partitioner = t5x.partitioning.PjitPartitioner(**config.partitioning)
    partitioner._logical_axis_rules += (("_null0", None),)
    partitioner._logical_axis_rules += (("_null1", None),)
    partitioner._logical_axis_rules += (("_null2", None),)
    partitioner._logical_axis_rules += (("classes", None),)

    # ------------------------------------
    # Create data loader
    # ------------------------------------
    data_loader_train, data_loader_val, data_loader_tags = build_dataloaders(
        config, partitioner
    )
    batched_tags = [d for d in data_loader_tags]  # 1000x80 or 1000x7

    steps_per_epoch = config.samples_per_epoch // config.batch_size  # for lr schedule

    # ------------------------------------
    # Create model
    # ------------------------------------
    model = models_flip.FLIP(config=config.model)
    init_batch = next(data_loader_train)
    p_init_fn, state_axes, state_shape = create_train_state(
        config, model, steps_per_epoch, partitioner, init_batch=init_batch
    )
    rng_init, rng = jax.random.split(rng)

    t5x.model_info.log_model_info(None, state_shape, partitioner)

    # ------------------------------------
    # Create checkpointer
    # ------------------------------------
    checkpointer = t5x.checkpoints.Checkpointer(
        train_state=state_shape,
        partitioner=partitioner,
        checkpoints_dir=workdir,
    )

    if config.resume_dir != "":
        state = ckp.restore_checkpoint(checkpointer, path=config.resume_dir)
    elif config.pretrain_dir != "":
        # raise NotImplementedError
        logging.info("Initializing train_state...")
        state = p_init_fn(rng_init)
        logging.info("Initializing train_state done.")
        logging.info("load pretrain")

        path = config.pretrain_dir
        step = t5x.checkpoints.latest_step(path)
        path_chkpt = (
            path if step is None else t5x.checkpoints.get_checkpoint_dir(path, step)
        )

        state = checkpointer.restore(
            path=path_chkpt,
            fallback_state=state.state_dict(),
            state_transformation_fns=[ckp.remove_optimizer_state, ckp.remove_pos_embed],
        )
    else:
        logging.info("Initializing train_state...")
        state = p_init_fn(rng_init)
        logging.info("Initializing train_state done.")

    t5x.model_info.log_state_info(state)

    # step_offset > 0 if restarting from checkpoint
    step_offset = int(state.step)
    logging.info("step_offset: {}".format(step_offset))

    # ------------------------------------------
    # Create partitioned eval_tags_step
    eval_step_fn = functools.partial(eval_tags_step, model=model, rng=rng)
    eval_axes = PartitionSpec("data", None)
    partitioned_eval_tags_step = partitioner.partition(
        eval_step_fn,
        in_axis_resources=(state_axes, partitioner.data_partition_spec),
        out_axis_resources=eval_axes,
    )
    # ------------------------------------------

    # ------------------------------------------
    # Create partitioned train_step
    train_step_fn = functools.partial(train_step, model=model, rng=rng)
    partitioned_train_step = partitioner.partition(
        train_step_fn,
        in_axis_resources=(state_axes, partitioner.data_partition_spec),
        out_axis_resources=(state_axes, None),
        donate_argnums=(0,),
    )
    # ------------------------------------------

    # ------------------------------------------
    # Create partitioned eval_step
    eval_step_fn = functools.partial(eval_step, model=model, rng=rng)
    eval_axes = None
    partitioned_eval_step = partitioner.partition(
        eval_step_fn,
        in_axis_resources=(state_axes, partitioner.data_partition_spec, None),
        out_axis_resources=eval_axes,
    )
    # ------------------------------------------

    # ------------------------------------------
    if config.eval_only:
        logging.info("Eval only...")
        summary = run_eval(
            state,
            batched_tags,
            partitioned_eval_tags_step,
            data_loader_val,
            partitioned_eval_step,
            config,
        )
        values = [f"{k}: {v:.6f}" for k, v in sorted(summary.items())]
        logging.info("eval: %s", ", ".join(values))
        return
    # ------------------------------------------

    train_metrics = []

    logging.info("Work dir: {}".format(workdir))
    train_metrics_last_t = time.time()
    logging.info("Initial compilation, this might take some minutes...")

    epoch_offset = (step_offset + 1) // steps_per_epoch
    step = epoch_offset * steps_per_epoch

    data_layout = partitioner.get_data_layout(config.batch_size)
    shard_id = data_layout.shard_id

    for epoch in range(epoch_offset, int(config.num_epochs)):
        # ------------------------------------------------------------
        # train one epoch (one "virtual" epoch)
        # ------------------------------------------------------------
        for i in range(steps_per_epoch):
            batch = next(data_loader_train)
            state, metrics = partitioned_train_step(state, batch)

            if epoch == epoch_offset and i == 0 and partitioner._num_partitions > 8:
                print_sanity_check(batch, shard_id)

            # normalize to IN1K epoch anyway
            epoch_1000x = int(step * config.batch_size / 1281167 * 1000)

            if epoch == epoch_offset and i == 0:
                logging.info("Initial compilation completed.")
                # log the time after compilation
                start_time = time.time()

            if config.get("log_every_steps"):
                train_metrics.append(metrics)
                if (step + 1) % config.log_every_steps == 0:
                    # Wait until computations are done before exiting
                    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
                    train_metrics = common_utils.get_metrics(
                        jax.tree_map(lambda x: jnp.reshape(x, (-1,)), train_metrics)
                    )
                    summary = {
                        f"train_{k}": float(v)
                        for k, v in jax.tree_map(
                            lambda x: x.mean(), train_metrics
                        ).items()
                    }
                    summary["steps_per_second"] = config.log_every_steps / (
                        time.time() - train_metrics_last_t
                    )

                    # to make it consistent with PyTorch log
                    summary["loss"] = summary["train_loss"]  # add extra name
                    summary["lr"] = summary.pop("train_learning_rate")  # rename
                    # step for tensorboard
                    summary["step_tensorboard"] = epoch_1000x

                    writer.write_scalars(step + 1, summary)
                    train_metrics = []
                    train_metrics_last_t = time.time()

            step += 1

        # ------------------------------------------------------------
        # finished one epoch: eval
        # ------------------------------------------------------------
        vis_every_epochs = config.vis_every_epochs
        if (epoch + 1) % vis_every_epochs == 0 or epoch == epoch_offset:
            summary = run_eval(
                state,
                batched_tags,
                partitioned_eval_tags_step,
                data_loader_val,
                partitioned_eval_step,
                config,
            )
            values = [f"{k}: {v:.6f}" for k, v in sorted(summary.items())]
            logging.info("eval epoch: %d, %s", epoch, ", ".join(values))

            # to make it consistent with PyTorch log
            summary[
                "step_tensorboard"
            ] = epoch  # step for tensorboard (no need to minus 1)
            writer.write_scalars(step + 1, summary)
            writer.flush()

        # ------------------------------------------------------------
        # finished one epoch: save
        # ------------------------------------------------------------
        if (
            (epoch + 1) % config.save_every_epochs == 0
            or epoch + 1 == int(config.num_epochs)
            or epoch == epoch_offset
        ):
            logging.info("Saving checkpoint: {}".format(workdir))
            checkpointer.save(state)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Elapsed time: {}".format(total_time_str))

    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    return state


def compute_encoded_tags(
    state,
    batched_tags,
    partitioned_eval_tags_step,
):
    # Encoding tags: no data-parallism across nodes
    logging.info("Encoding tags...")
    encoded_tags = []
    for i, tags_batch in enumerate(batched_tags):
        z_txt = partitioned_eval_tags_step(state, tags_batch)
        encoded_tags.append(z_txt)
        if i % 100 == 0:
            logging.info("{} / {}".format(i, len(batched_tags)))
    encoded_tags = jnp.concatenate(encoded_tags, axis=0)  # type: DeviceArray

    # ----------------
    # average multiple templates
    encoded_tags = encoded_tags.reshape(
        [1000, -1, encoded_tags.shape[-1]]
    )  # [1000, 7, 512]
    encoded_tags = encoded_tags.mean(axis=1)
    encoded_tags /= jnp.linalg.norm(encoded_tags, axis=-1, keepdims=True) + 1e-8
    assert encoded_tags.shape[0] == 1000
    # ----------------

    logging.info("Encoding tags done: {}".format(encoded_tags.shape))
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
            logging.info(
                "{} / {}, shape: {}".format(
                    i, steps_per_eval, eval_batch["image"].shape
                )
            )

    eval_metrics = jax.device_get(eval_metrics)
    eval_metrics = jax.tree_map(lambda *args: np.concatenate(args), *eval_metrics)

    valid = np.where(eval_metrics["label"] >= 0)
    eval_metrics.pop("label")
    eval_metrics = jax.tree_util.tree_map(lambda x: x[valid], eval_metrics)

    toc = time.time() - tic
    logging.info(
        "Eval time: {}, {} steps, {} samples".format(
            str(datetime.timedelta(seconds=int(toc))),
            steps_per_eval,
            len(eval_metrics["test_acc1"]),
        )
    )

    summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
    return summary
