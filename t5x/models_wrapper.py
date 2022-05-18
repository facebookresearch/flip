
import abc
import functools
from typing import Any, Callable, Mapping, MutableMapping, Optional, Tuple, Union

import clu.metrics as clu_metrics
from flax import core as flax_core
from flax import linen as nn
from flax.core import scope as flax_scope
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
# import seqio
import optax
# from t5x import decoding
# from t5x import losses
# from t5x import metrics as metrics_lib
from t5x import optimizers
import tensorflow as tf
import typing_extensions

Array = Union[np.ndarray, jnp.ndarray, jax.pxla.ShardedDeviceArray, tf.Tensor]
# MetricsMap = metrics_lib.MetricsMap
MetricsMap = Any
PyTreeDef = type(jax.tree_structure(None))


class BaseModel(abc.ABC):
  """Abstract base class for models.

  Wraps a flax module to provide a basic interface for computing loss,
  evaluation metrics, prediction, and scoring.

  Subclasses must implement the abstract methods. Any additional arguments added
  to these methods must have defaults or be bound at run time to fit the
  interface expected by the standard training, inference, and evaluation
  functions.
  """

  def __init__(self, optimizer_def: optimizers.OptimizerDefType):
    # TODO(jbulian): Move the optimizer out of the model and make it a training
    #                parameter.
    self.optimizer_def = optimizer_def

  @abc.abstractmethod
  def loss_fn(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.random.KeyArray],
  ):
    """Computes loss and metrics.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      dropout_rng: rng to use for dropout, or None for deterministic mode.

    Returns:
      loss: the loss computed for the given inputs and parameters.
      aux:
        weight_sum: sum of the per-token weights applied to the loss.
        metrics: a mapping of metrics computed for this batch.
    """
    pass


class BaseTransformerModel(BaseModel):
  """Abstract base class for Transformer models.

  Subclasses must implement `predict_batch_with_aux`, `score_batch`,
  `get_initial_variables` from `BaseModel` as well as `_compute_logits`.
  """

  def __init__(
      self,
      module: nn.Module,
      optimizer_def: optimizers.OptimizerDefType,
  ):
    self.module = module
    super().__init__(optimizer_def=optimizer_def)

  def _compute_logits(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.random.KeyArray] = None) -> jnp.ndarray:
    """Computes logits via a forward pass of the model."""
    logits = self.module.apply(
        {'params': params,}, # {'params': params, **flax_mutables},
        inputs=batch['image'],
        mutable=False, # mutable=flax_mutables.keys(),
        rngs=dict(dropout=dropout_rng),
        train=True)
    return logits


  def loss_fn(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.random.KeyArray],
  ):
    """Loss function used for training with a cross-entropy loss."""
    logits = self._compute_logits(params, batch, dropout_rng)

    loss = cross_entropy_loss(logits, batch['label_one_hot'])
    return loss, logits


def cross_entropy_loss(logits, labels_one_hot):
  xentropy = optax.softmax_cross_entropy(logits=logits, labels=labels_one_hot)
  return jnp.mean(xentropy)
