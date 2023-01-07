# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from absl import logging

import t5x.checkpoints


# --------------------------------------------
# checkpoint interfaces
# --------------------------------------------
def restore_checkpoint(checkpointer, path):
    step = t5x.checkpoints.latest_step(
        path
    )  # try to load the latest checkpoint if not specified
    path_chkpt = (
        path if step is None else t5x.checkpoints.get_checkpoint_dir(path, step)
    )
    state = checkpointer.restore(path=path_chkpt)
    return state


def remove_optimizer_state(ckpt_optimizer_state, optimizer_state):
    logging.info("pop state")
    ckpt_optimizer_state.pop("state")
    return ckpt_optimizer_state


def remove_pos_embed(ckpt_optimizer_state, optimizer_state):
    if (
        "posembed_encoder" in ckpt_optimizer_state["target"]["img_encoder"]
        and "posembed_encoder" in optimizer_state["target"]["img_encoder"]
    ):
        shape_ckpt = ckpt_optimizer_state["target"]["img_encoder"]["posembed_encoder"][
            "pos_embedding"
        ]["metadata"]["shape"]
        shape_opt = list(
            optimizer_state["target"]["img_encoder"]["posembed_encoder"][
                "pos_embedding"
            ].shape
        )
        if not (shape_ckpt == shape_opt):
            logging.info("Removing pre-trained posembed_encoder.")
            ckpt_optimizer_state["target"]["img_encoder"].pop("posembed_encoder")
    return ckpt_optimizer_state
