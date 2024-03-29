# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# --------------------------------------------------------
# References:
# The code is adapted and modified from https://github.com/google-research/t5x/tree/main/t5x
# LICENSE: https://github.com/google-research/t5x/blob/2a62e14fd2806a28c8b24c7674fdd5423aa95e3d/LICENSE
# --------------------------------------------------------


"""Checkpoint helper functions for managing checkpoints.

Supports marking checkpoints as pinned to exclude them from the checkpointer
removal process.
"""

import os

from absl import logging

from tensorflow.io import gfile

# PINNED file in the checkpoint directory indicates that the checkpoint should
# not be removed during the automatic pruning of old checkpoints.
_PINNED_CHECKPOINT_FILENAME = "PINNED"


def pinned_checkpoint_filepath(ckpt_dir: str) -> str:
    """Full path of the pinned checkpoint file."""
    return os.path.join(ckpt_dir, _PINNED_CHECKPOINT_FILENAME)


def is_pinned_checkpoint(ckpt_dir: str) -> bool:
    """Returns whether the checkpoint is pinned, and should NOT be removed."""
    pinned_ckpt_file = pinned_checkpoint_filepath(ckpt_dir)
    if gfile.exists(pinned_ckpt_file):
        return True
    return False


def pin_checkpoint(ckpt_dir: str, txt: str = "1") -> None:
    """Pin a checkpoint so it does not get deleted by the normal pruning process.

    Creates a PINNED file in the checkpoint directory to indicate the checkpoint
    should be excluded from the deletion of old checkpoints.

    Args:
      ckpt_dir: The checkpoint step dir that is to be always kept.
      txt: Text to be written into the checkpoints ALWAYS_KEEP me file.
    """
    pinned_ckpt_file = pinned_checkpoint_filepath(ckpt_dir)
    with gfile.GFile(pinned_ckpt_file, "w") as f:
        logging.debug("Write %s file : %s.", pinned_ckpt_file, txt)
        f.write(txt)


def unpin_checkpoint(ckpt_dir: str) -> None:
    """Removes the pinned status of the checkpoint so it is open for deletion."""
    if not is_pinned_checkpoint(ckpt_dir):
        logging.debug("%s is not PINNED. Nothing to do here.", ckpt_dir)
        return
    try:
        pinned_ckpt_file = pinned_checkpoint_filepath(ckpt_dir)
        logging.debug("Remove %s file.", pinned_ckpt_file)
        gfile.rmtree(pinned_ckpt_file)
    except IOError:
        logging.exception("Failed to unpin %s", ckpt_dir)


def remove_checkpoint_dir(ckpt_dir: str) -> None:
    """Removes the checkpoint dir if it is not pinned."""
    if not is_pinned_checkpoint(ckpt_dir):
        logging.info("Deleting checkpoint: %s", ckpt_dir)
        gfile.rmtree(ckpt_dir)
    else:
        logging.info("Keeping pinned checkpoint: %s", ckpt_dir)
