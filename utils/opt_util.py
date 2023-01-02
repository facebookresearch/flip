import jax.numpy as jnp

import tree as nest

from typing import Tuple, Any


# ---------------------------------------------------------
# exclude wd
# ---------------------------------------------------------
def filter_bias_and_norm(path: Tuple[Any], val: jnp.ndarray):
    """Filter to exclude biases and normalizations weights."""
    del val
    if path[-1] == "bias" or path[-1] == "scale":
        return False
    return True


def filter_cls_and_posembed(path: Tuple[Any], val: jnp.ndarray):
    """Filter to exclude cls token and pos emb."""
    del val
    name = ".".join(path)
    if "pos_embedding" in name or path[-1] == "cls":
        return False
    return True


def filter_posembed(path: Tuple[Any], val: jnp.ndarray):
    """Filter to exclude pos emb."""
    del val
    name = ".".join(path)
    if "pos_embedding" in name:
        return False
    return True


# ---------------------------------------------------------
# freeze parameters (e.g CLIP Encoder)
# ---------------------------------------------------------
def filter_freeze_keys(path: Tuple[Any], val: jnp.ndarray, keys=[]):
    """Filter to exclude img feature extractor"""
    del val
    name = ".".join(path)
    for k in keys:
        if k in name:
            return False

    return True


# ---------------------------------------------------------
# the entrance function:
# ---------------------------------------------------------
def filter_parameters(params, filter_fn):
    """Filter the params based on filter_fn."""
    params_to_filter = nest.map_structure_with_path(filter_fn, params)
    return params_to_filter
