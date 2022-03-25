import tensorflow as tf
from clu.metric_writers.summary_writer import SummaryWriter


def write_scalars(self, step: int, scalars):
  """ Revise write_scalars to support epoch_1000x
  """
  if 'step_tensorboard' in scalars:
      step = scalars['step_tensorboard']
  with self._summary_writer.as_default():
    for key, value in scalars.items():
      tf.summary.scalar(key, value, step=step)
SummaryWriter.write_scalars = write_scalars
