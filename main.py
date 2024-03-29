# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
# https://github.com/google/flax/tree/main/examples/imagenet


from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
from ml_collections import config_flags
import tensorflow as tf
import time
import warnings

import train
from utils import logging_util

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], "GPU")

    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(
        f"process_index: {jax.process_index()}, "
        f"process_count: {jax.process_count()}"
    )
    platform.work_unit().create_artifact(
        platform.ArtifactType.DIRECTORY, FLAGS.workdir, "workdir"
    )

    logging.info(FLAGS.config)

    train.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
    if jax.process_count() > 1:
        time.sleep(2)  # wait for all the tf warnings
    logging_util.verbose_off()
    logging_util.set_time_logging(logging)
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
