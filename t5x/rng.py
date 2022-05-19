import functools
import jax
from jax import prng


def set_hardware_rng_ops():
  """Enable JAX Custom PRNG extension."""
  jax.config.update('jax_enable_custom_prng', True)
  # Use only fast TPU hardware PRNG with iterated-hash "split" substitute.
  # Expected to be deterministic for a fixed partitioning.
  # Monkey-patch JAX PRNGKey to use unsafe_rbg_prng_impl
  # TODO(levskaya): replace with jax global config option once we debug it.
  rbg_prng_key = functools.partial(prng.seed_with_impl,
                                   prng.unsafe_rbg_prng_impl)
  jax.random.PRNGKey = rbg_prng_key
  jax._src.random.PRNGKey = rbg_prng_key  # pylint: disable=protected-access
