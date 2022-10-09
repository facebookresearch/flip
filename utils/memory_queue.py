import functools
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

import flax.linen as nn

import t5x

from utils import initializers_util



# # queue is not batched; ids is batched
# gather = jax.vmap(lambda queue, ids: queue[ids], in_axes=(None, 0), out_axes=0)


# initialize queue
def queue_features_init(shape):
  x = jax.random.normal(jax.random.PRNGKey(0), shape)
#   l2norm = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1.e-12)
#   x /= l2norm
  x /= jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
  return x


class MemQueue(nn.Module):
  """Online memory queue during training.
  """
  queue_size: int
  queue_dim: int

  def setup(self):
    K = self.queue_size
    D = self.queue_dim

    # create the queue
    #queue_features = self.variable('momqueue_vars', 'queue_features', queue_features_init, (K, D))
    #queue_ptr = self.variable('momqueue_vars', 'queue_ptr', lambda s: jnp.zeros(s, jnp.int32), ())

    self.queue_features = t5x.layers.variable_with_axes(
        'memqueue_vars', 'queue_features', queue_features_init, (K, D), axes=('batch', "_null1")
    )
    self.queue_ptr = t5x.layers.variable_with_axes(
        'memqueue_vars', 'queue_ptr', lambda s: jnp.zeros(s, jnp.int32), 
        (1,), axes=('_null0',)
        #(),
    )
  
  def update_queue(self, features):
    batch_size = features.shape[0]
    assert self.queue_size >= batch_size
    nd = min(jax.device_count(), batch_size)
    size_per_device = self.queue_size // nd

    inds0 = jnp.arange(batch_size // nd) + self.queue_ptr.value
    inds0 = jnp.mod(inds0, size_per_device)
    # inds0 = jnp.sort(inds0)
    inds = jnp.concatenate([inds0 + i * size_per_device for i in range(nd)], axis=0)
    # from jax.experimental.host_callback import call
    # call(lambda x: print(x.shape, x), inds)

    self.queue_features.value = self.queue_features.value.at[inds].set(features)
    self.queue_ptr.value = (self.queue_ptr.value + batch_size // nd) % size_per_device

    # inds = jnp.arange(batch_size)
    # self.queue_features.value = self.queue_features.value.at[inds].set(features)

    return 

  def get_queue(self):
    return self.queue_features.value
