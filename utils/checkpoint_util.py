from absl import logging

import jax
import flax
from flax.training import checkpoints
import jax.numpy as jnp

import jax.tree_util as tu

def load_from_pretrain(state, pretrain_dir):
  state_load = checkpoints.restore_checkpoint(pretrain_dir, target=None)
  params_load = flax.core.freeze(state_load.pop('params'))  # match the type of state.params

  variables_load = state_load.pop('variables')
  assert variables_load == {}  # no state variables in ViT (no BatchNorm)

  del state_load

  missing_keys = set(state.params.keys()) - set(params_load.keys())
  load_keys = set(state.params.keys()) - missing_keys
  ignored_keys = set(params_load.keys()) - load_keys - missing_keys

  params = {k: state.params[k] for k in missing_keys}

  for k in load_keys:
    p_state, p_load = state.params[k], params_load[k]
    assert len(tu.tree_leaves(p_state)) == len(tu.tree_leaves(p_load))
    assert tu.tree_structure(p_state) == tu.tree_structure(p_load)
    verify = tu.tree_multimap(lambda x, y: (x.shape == y.shape), p_state, p_load)
    verify = jnp.all(jnp.array(tu.tree_leaves(verify)))
    assert verify, 'Not matching: {}\n{}'.format(k,
    tu.tree_multimap(
        lambda x, y: (x.shape, y.shape) if (x.shape != y.shape) else None,
        p_state, p_load)
    )
    params[k] = p_load
  
  params = flax.core.freeze(params)
  assert tu.tree_structure(params) == tu.tree_structure(state.params)

  logging.info('Loaded keys: {}'.format(load_keys))
  logging.info('Missing keys: {}'.format(missing_keys))
  logging.info('Ignored keys: {}'.format(ignored_keys))

  return state.replace(params=params)