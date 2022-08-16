"""General utility functions for t5x."""
import contextlib
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, Type, Union

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
from t5x import partitioning
from t5x import state_utils
from t5x import train_state as train_state_lib
import tensorflow as tf
from tensorflow.io import gfile

from utils import logging_util


Array = Union[np.ndarray, jnp.ndarray, jax.pxla.ShardedDeviceArray, tf.Tensor]
PyTreeDef = type(jax.tree_structure(None))
PartitionSpec = partitioning.PartitionSpec
DType = Union[np.dtype, type(jnp.bfloat16)]
Shape = Tuple[int, ...]

# -----------------------------------------------------------------------------
# Logging utility functions
# -----------------------------------------------------------------------------
def log_model_info(log_file: Optional[str],
                   full_train_state: train_state_lib.TrainState,
                   partitioner: partitioning.BasePartitioner):
  """Log the variable shapes information and optionally write it to a file."""
  # Only write logs on host 0.
  if jax.process_index() != 0:
    return

  state_dict = full_train_state.state_dict()
  total_num_params = jax.tree_util.tree_reduce(
      np.add, jax.tree_map(np.size, state_dict['target']))
  total_num_states = jax.tree_util.tree_reduce(
      np.add, jax.tree_map(np.size, state_dict['state']))

  logical_axes = partitioner.get_logical_axes(full_train_state).state_dict()

  mesh_axes = jax.tree_map(
      lambda x: tuple(x) if x is not None else None,
      partitioner.get_mesh_axes(full_train_state).state_dict())

  def _log_info_and_write_to_file(writer, format_str, *args):
    logging.info(format_str, *args)
    if writer is not None:
      writer.write(format_str % args + '\n')

  with contextlib.ExitStack() as stack:
    writer = stack.enter_context(gfile.GFile(
        log_file, 'w')) if log_file is not None else None

    mesh_shape = partitioner._local_chunker.global_mesh.shape  # e.g., OrderedDict([('data', 2), ('model', 4)])
    
    # Log params
    def _log_variable(name: str, arr: Optional[np.ndarray],
                      logical_axes: Optional[partitioning.AxisNames],
                      mesh_axes: Optional[partitioning.PartitionSpec]):
      # Log nothing on empty dict leaves, which occur with optax EmptyState().
      if isinstance(arr, dict) and not arr:
        return
      if arr is None:
        _log_info_and_write_to_file(writer, 'Variable    %-80s None', name)
        return
      if logical_axes is None or len(logical_axes) != len(arr.shape):
        shape_str = str(arr.shape)
      else:
        shape_str = '({})'.format(', '.join(
            f'{name}={dimension}'
            for name, dimension in zip(logical_axes, arr.shape)))
      # _log_info_and_write_to_file(
      #     writer, '%-96s %-20s %-40s %s',
      #     name, arr.size, shape_str, mesh_axes)
      arr_size = '{:,d}'.format(arr.size)
      logging.info('{:108s} {:>16s} {:48s} {}'.format(name, arr_size, str(shape_str), str(mesh_axes)))

      # kaiming: hack, do the sanity check here
      if len(arr.shape) > 0:
        for dim, ax in zip(arr.shape, mesh_axes):
          if ax is not None:
            assert dim % mesh_shape[ax] == 0

    logging_util.set_time_logging_short(logging)  # make the logging shorter

    jax.tree_map(
        _log_variable,
        state_utils.get_name_tree(state_dict['target'], keep_empty_nodes=True),
        state_dict['target'], logical_axes['target'], mesh_axes['target'])

    # Add a blank line between params and states.
    _log_info_and_write_to_file(writer, '')

    jax.tree_map(
        _log_variable,
        state_utils.get_name_tree(state_dict['state'], keep_empty_nodes=True),
        state_dict['state'], logical_axes['state'], mesh_axes['state'])
    
    logging_util.set_time_logging(logging)  # restore the logging

    _log_info_and_write_to_file(writer, 'Total number of parameters (1e6): %.6f',
                                total_num_params / 1e6)
    _log_info_and_write_to_file(writer, 'Total number of parameter_states (1e6): %.6f',
                                total_num_states / 1e6)

    expected_memory = (total_num_params + total_num_states) * 4
    _log_info_and_write_to_file(writer, 'Estimated model memory with all float32 (G): %.6f',
                                expected_memory / 1024 / 1024 / 1024)


def log_state_info(real_train_state: train_state_lib.TrainState,):
  """Log the variable shapes information and optionally write it to a file."""
  # Only write logs on host 0.
  if jax.process_index() != 0:
    return

  state_dict = real_train_state.state_dict()

  def parse_sharding(x):
    return np.array([i.chunks[0] if hasattr(i, 'chunks') else 1 for i in x.sharding_spec.sharding] if hasattr(x, 'sharding_spec') else [1,])

  shards = jax.tree_map(parse_sharding, state_dict)
  splits = jax.tree_map(lambda x: np.prod(x), shards)

  sizes = jax.tree_map(lambda x: x.size if hasattr(x, 'size') else 1, state_dict)

  jax.tree_util.tree_reduce(np.add, sizes['target']) / 1e6
  jax.tree_util.tree_reduce(np.add, sizes['state']) / 1e6

  types = jax.tree_map(lambda x: x.dtype if hasattr(x, 'dtype') else None, state_dict)
  bytes = jax.tree_map(lambda x: 4 if x == jnp.float32 else (2 if x == jnp.bfloat16 else 4), types)
  memory = jax.tree_map(lambda x, y, z: x * y / z, sizes, bytes, splits)

  total_mem_params = jax.tree_util.tree_reduce(np.add, memory['target'])
  total_mem_states = jax.tree_util.tree_reduce(np.add, memory['state'])

  logging.info('Est. memory of parameters per device (G): %.6f', total_mem_params / 1024 / 1024 / 1024)
  logging.info('Est. memory of parameter_states per device (G): %.6f', total_mem_states / 1024 / 1024 / 1024)
  logging.info('Est. memory of both per device (G): %.6f', (total_mem_params + total_mem_states) / 1024 / 1024 / 1024)