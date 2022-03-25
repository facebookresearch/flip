import logging

from termcolor import colored

def set_time_logging(logger):
  prefix = "[%(asctime)s.%(msecs)03d %(levelname)s:%(filename)s:%(lineno)d] "
  str = colored(prefix, "green") + '%(message)s'
  logger.get_absl_handler().setFormatter(logging.Formatter(str, datefmt='%m%d %H:%M:%S'))
