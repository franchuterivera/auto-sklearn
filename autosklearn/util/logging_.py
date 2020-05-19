# -*- encoding: utf-8 -*-
import logging
import logging.config
import os

from memory_profiler import profile
import yaml


@profile
def setup_logger(output_file=None, logging_config=None):
    # logging_config must be a dictionary object specifying the configuration
    # for the loggers to be used in auto-sklearn.
    if logging_config is not None:
        if output_file is not None:
            logging_config['handlers']['file_handler']['filename'] = output_file
        logging.config.dictConfig(logging_config)
    else:
        with open(os.path.join(os.path.dirname(__file__), 'logging.yaml'),
                  'r') as fh:
            logging_config = yaml.safe_load(fh)
        if output_file is not None:
            logging_config['handlers']['file_handler']['filename'] = output_file
        logging.config.dictConfig(logging_config)


@profile
def _create_logger(name):
    return logging.getLogger(name)


@profile
def get_logger(name):
    logger = PickableLoggerAdapter(name)
    return logger


class PickableLoggerAdapter(object):

    @profile
    def __init__(self, name):
        self.name = name
        self.logger = _create_logger(name)

    @profile
    def __getstate__(self):
        """
        Method is called when pickle dumps an object.

        Returns
        -------
        Dictionary, representing the object state to be pickled. Ignores
        the self.logger field and only returns the logger name.
        """
        return {'name': self.name}

    @profile
    def __setstate__(self, state):
        """
        Method is called when pickle loads an object. Retrieves the name and
        creates a logger.

        Parameters
        ----------
        state - dictionary, containing the logger name.

        """
        self.name = state['name']
        self.logger = _create_logger(self.name)

    @profile
    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    @profile
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    @profile
    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    @profile
    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    @profile
    def exception(self, msg, *args, **kwargs):
        self.logger.exception(msg, *args, **kwargs)

    @profile
    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)

    @profile
    def log(self, level, msg, *args, **kwargs):
        self.logger.log(level, msg, *args, **kwargs)

    @profile
    def isEnabledFor(self, level):
        return self.logger.isEnabledFor(level)
