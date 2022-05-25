import logging as _logging
from absl import logging

import jax
from jax.experimental import multihost_utils

from termcolor import colored
import time

def set_time_logging(logger):
  pid = jax.process_index()
  prefix = "[p{:02d} %(asctime)s.%(msecs)03d %(levelname)s:%(filename)s:%(lineno)d] ".format(pid)
  str = colored(prefix, "green") + '%(message)s'
  logger.get_absl_handler().setFormatter(_logging.Formatter(str, datefmt='%m%d %H:%M:%S'))


def set_time_logging_short(logger):
  pid = jax.process_index()
  prefix = "[p{:02d} %(asctime)s] ".format(pid)
  str = colored(prefix, "green") + '%(message)s'
  logger.get_absl_handler().setFormatter(_logging.Formatter(str, datefmt='%m%d %H:%M:%S'))


def verbose_on():
  logging.set_verbosity(logging.INFO)  # show all processes


def verbose_off():
  if not (jax.process_index() == 0):  # not first process
    logging.set_verbosity(logging.ERROR)  # disable info/warning


def sync_and_delay(delay=None):
  # Block all hosts until directory is ready.
  multihost_utils.sync_global_devices(f'logging')
  if delay is None:
    delay = jax.process_index() * 0.1
  time.sleep(delay)