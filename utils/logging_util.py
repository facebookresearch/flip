import logging as _logging
from absl import logging

import jax

from termcolor import colored


def set_time_logging(logger):
  prefix = "[%(asctime)s.%(msecs)03d %(levelname)s:%(filename)s:%(lineno)d] "
  str = colored(prefix, "green") + '%(message)s'
  logger.get_absl_handler().setFormatter(_logging.Formatter(str, datefmt='%m%d %H:%M:%S'))


def verbose_on():
  logging.set_verbosity(logging.INFO)  # show all processes


def verbose_off():
  if not (jax.process_index() == 0):  # not first process
    logging.set_verbosity(logging.ERROR)  # disable info/warning
