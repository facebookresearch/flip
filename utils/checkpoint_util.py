import jax
import flax
from flax.training import checkpoints
import jax.numpy as jnp

def load_from_pretrain(state, pretrain_dir, ignored_names={'head'}):
  state_load = checkpoints.restore_checkpoint(pretrain_dir, target=None)
  params_load = flax.core.freeze(state_load.pop('params'))  # match the type of state.params

  variables_load = state_load.pop('variables')
  assert variables_load == {}  # no state variables in ViT (no BatchNorm)

  del state_load

  params = {k: state.params[k] for k in ignored_names}

  for k in state.params.keys():
    if k not in ignored_names:
      p_state, p_load = state.params[k], params_load[k]
      assert len(jax.tree_util.tree_leaves(p_state)) == len(jax.tree_util.tree_leaves(p_load))
      assert jax.tree_util.tree_structure(p_state) == jax.tree_util.tree_structure(p_load)
      verify = jax.tree_multimap(lambda x, y: (x.shape == y.shape), p_state, p_load)
      verify = jnp.all(jnp.array(jax.tree_util.tree_leaves(verify)))
      assert verify, 'Not matching: {}\n{}'.format(k,
        jax.tree_multimap(lambda x, y: (x.shape, y.shape) if (x.shape != y.shape) else None, p_state, p_load)
      )
      params[k] = p_load
  
  params = flax.core.freeze(params)
  assert jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(state.params)
  
  return state.replace(params=params)