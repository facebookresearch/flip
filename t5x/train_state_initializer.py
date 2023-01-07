# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# --------------------------------------------------------
# References:
# The code is adapted and modified from https://github.com/google-research/t5x/tree/main/t5x
# LICENSE: https://github.com/google-research/t5x/blob/2a62e14fd2806a28c8b24c7674fdd5423aa95e3d/LICENSE
# --------------------------------------------------------


from absl import logging
import jax
import jax.numpy as jnp
import optax
import functools

import t5x.train_state as train_state_lib
import t5x.optimizers
from t5x import state_utils

from utils import opt_util
from utils import adamw


def init_fn(rng, init_batch, model):
    variables = model.init(
        {"params": rng, "dropout": jax.random.PRNGKey(0)}, init_batch, train=True
    )
    return variables


def init_shapes(rng, init_batch, model):
    # input_shape = (1, image_size, image_size, 3)
    init = functools.partial(model.init, train=True)
    variables_shape = jax.eval_shape(
        init, {"params": rng, "dropout": jax.random.PRNGKey(0)}, init_batch
    )
    return variables_shape


def create_learning_rate_fn(config, base_learning_rate: float, steps_per_epoch: int):
    """Create learning rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=config.warmup_abs_lr,
        end_value=base_learning_rate,
        transition_steps=config.warmup_epochs * steps_per_epoch,
    )
    cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=cosine_epochs * steps_per_epoch,
        alpha=config.min_abs_lr / base_learning_rate,
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[config.warmup_epochs * steps_per_epoch],
    )
    return schedule_fn


def create_optimizer(config, params_names, steps_per_epoch):
    """create optimizer"""
    # create the lr schedule function
    abs_learning_rate = config.learning_rate * config.batch_size / 256.0
    learning_rate_fn = create_learning_rate_fn(
        config, abs_learning_rate, steps_per_epoch
    )

    if config.opt_type in {"adamw", "adarows"}:
        # optional: exclude some wd
        mask = None
        if config.exclude_wd:
            mask = jax.tree_util.tree_map(
                lambda x, y: bool(x and y),
                opt_util.filter_parameters(params_names, opt_util.filter_bias_and_norm),
                opt_util.filter_parameters(
                    params_names, opt_util.filter_posembed
                ),  # Note: we must exclude posembed wd in adamw
            )
        logging.info("Apply wd: {}".format(state_utils.str_flatten_dict(mask)))

        opt = getattr(adamw, config.opt_type)  # optax.adamw

        opt = t5x.optimizers.wrap_optax_optimizer(opt)
        opt = opt(
            learning_rate=learning_rate_fn,
            **config.opt,
            mask=mask,
            mu_dtype=getattr(jnp, config.opt_mu_dtype),
        )
        opt.metric_learning_rate_fn = learning_rate_fn  # hack for metric

    else:
        raise NotImplementedError

    return opt


def create_train_state(config, model, steps_per_epoch, partitioner, init_batch):
    """Create initial training state."""

    init_batch = jax.tree_map(lambda x: x[:2], init_batch)  # set batch=2 for init

    rng = jax.random.PRNGKey(0)  # for shape reference only
    # create optimizer first
    params_shapes = init_shapes(rng, init_batch, model)
    opt = create_optimizer(config, params_shapes["params"], steps_per_epoch)

    # ---------------------------------------------------------------------------
    def initialize_train_state(rng_init):
        # split rng for init and for state
        initial_variables = init_fn(rng=rng_init, init_batch=init_batch, model=model)
        if opt:
            return train_state_lib.FlaxOptimTrainState.create(opt, initial_variables)
        return train_state_lib.InferenceState.create(initial_variables)

    train_state_shape = jax.eval_shape(initialize_train_state, rng_init=rng)
    train_state_axes = partitioner.get_mesh_axes(train_state_shape)

    p_init_fn = partitioner.partition(
        initialize_train_state,
        in_axis_resources=None,
        out_axis_resources=train_state_axes,
    )

    return p_init_fn, train_state_axes, train_state_shape
