import jax.numpy as jnp

import tree as nest

from typing import Tuple, Any

# reference: https://github.com/rwightman/efficientnet-jax/blob/6e343d5ff4e02f4500c3d979cb75bb0f33e2c601/jeffnet/common/optim/lars.py#L36


# ---------------------------------------------------------
# rescale layers
# ---------------------------------------------------------
def layer_rescale(path: Tuple[Any], val: jnp.ndarray):
    """Rescale the last layer of each block by layer_id."""
    del val
    # path[0] = 'Transformer'
    if len(path) > 3 and path[1].startswith("encoderblock_") and path[-1] == 'kernel':
        if (path[-3] == 'MultiHeadDotProductAttention_0' and path[-2] == 'out') or \
                (path[-3] == 'MlpBlock_0' and path[-2] == 'Dense_1'):
            layer_id = path[1][len("encoderblock_"):]  # remove prefix
            layer_id = int(layer_id) + 1
            rescale = (2.0 * layer_id) ** -.5
            return rescale
    return 1.


# ---------------------------------------------------------
# exclude wd
# ---------------------------------------------------------
def filter_bias_and_norm(path: Tuple[Any], val: jnp.ndarray):
    """Filter to exclude biases and normalizations weights."""
    del val
    if path[-1] == "bias" or path[-1] == 'scale':
        return False
    return True


def filter_cls_and_posembed(path: Tuple[Any], val: jnp.ndarray):
    """Filter to exclude cls token and pos emb."""
    del val
    name = '.'.join(path)
    if 'pos_embedding' in name or 'cls' == name:
        return False
    return True


# ---------------------------------------------------------
# the entrance function:
# ---------------------------------------------------------
def filter_parameters(params, filter_fn):
    """Filter the params based on filter_fn."""
    params_to_filter = nest.map_structure_with_path(filter_fn, params)
    return params_to_filter
