# -*- encoding: utf-8 -*-
import logging
import logging.config
import logging.handlers
import multiprocessing
import os
import pickle
import random
import select
import socketserver
import struct
import threading
from typing import Any, Dict, Optional, Type

import yaml


def setup_logger(
    output_dir: str,
    filename: Optional[str] = None,
    distributedlog_filename: Optional[str] = None,
    logging_config: Optional[Dict] = None,
) -> None:
    # logging_config must be a dictionary object specifying the configuration
    # for the loggers to be used in auto-sklearn.
    if logging_config is None:
        with open(os.path.join(os.path.dirname(__file__), 'logging.yaml'), 'r') as fh:
            logging_config = yaml.safe_load(fh)

    if filename is None:
        filename = logging_config['handlers']['file_handler']['filename']
    logging_config['handlers']['file_handler']['filename'] = os.path.join(
        output_dir, filename
    )

    if distributedlog_filename is None:
        distributedlog_filename = logging_config['handlers']['distributed_logfile']['filename']
    logging_config['handlers']['distributed_logfile']['filename'] = os.path.join(
            output_dir, distributedlog_filename
        )

    # Applying the configuration is expensive, because logging.config.dictConfig
    # reconstruct the logging singletons each time it is called. We only call it when
    # needed, and to do so, we created a initialized attributed to control if a logger
    # was created
    if not is_logging_config_applied(logging_config):
        logging.config.dictConfig(logging_config)
        for logger_name in list(logging_config['loggers'].keys()) + ['root']:
            if logger_name == 'root':
                logger = logging.getLogger()
            else:
                logger = logging.getLogger(logger_name)
            setattr(logger, 'initialized', True)


def is_logging_config_applied(logging_config: Dict) -> bool:
    """
    This functions check if the provided logging config is already applied to the environment.
    if it is not the case, it returns false.

    The motivation towards this checking is that in multiprocessing the loggers might be lost,
    because a new worker in a new node might not have the logger configuration setup.

    Parameters
    ----------
    logging_config: (Dict)
        A logging configuration following the format specified in
        https://docs.python.org/3/library/logging.config.html

    Returns
    -------
    (bool)
        True if a configuration has already been applied to the environment
    """

    # The logging config is a dictionary with
    # dict_keys(['version', 'disable_existing_loggers', 'formatters', 'handlers',
    # 'root', 'loggers']) . There are 2 things to check. Whether the logger changed
    # or whether the handlers changed

    # Check the loggers
    for logger_name in list(logging_config['loggers'].keys()) + ['root']:

        if logger_name == 'root':
            logger = logging.getLogger()
        else:
            logger = logging.getLogger(logger_name)

        # Checking for the contents of the logging config is not feasible as the
        # logger uses a hierarchical structure, so depending on where this function is
        # called, the logger_name requires the full hierarchy (like autosklearn.automl)
        # But because loggers are singletons we can rely on the initialized attribute we
        # set on them on creation
        if not hasattr(logger, 'initialized'):
            return False
    return True


def _create_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class PicklableClientLogger(object):

    def __init__(self, output_dir: str, name: str, host: str, port: int,
                 filename: Optional[str], logging_config: Optional[Dict]):
        self.output_dir = output_dir
        self.name = name
        self.host = host
        self.port = port
        self.filename = filename
        self.logging_config = logging_config
        self.logger = _get_named_client_logger(
            output_dir=self.output_dir,
            host=self.host,
            port=self.port,
            logging_config=self.logging_config,
            filename=self.filename,
        )

    def __getstate__(self) -> Dict[str, Any]:
        """
        Method is called when pickle dumps an object.

        Returns
        -------
        Dictionary, representing the object state to be pickled. Ignores
        the self.logger field and only returns the logger name.
        """
        return {'name': self.name, 'host': self.host, 'port': self.port,
                'filename': self.filename,
                'output_dir': self.output_dir, 'logging_config': self.logging_config}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Method is called when pickle loads an object. Retrieves the name and
        creates a logger.

        Parameters
        ----------
        state - dictionary, containing the logger name.

        """
        self.name = state['name']
        self.host = state['host']
        self.port = state['port']
        self.output_dir = state['output_dir']
        self.logging_config = state['logging_config']
        self.filename = state['filename']
        self.logger = _get_named_client_logger(
            output_dir=self.output_dir,
            host=self.host,
            port=self.port,
            logging_config=self.logging_config,
            filename=self.filename,
        )

    def format(self, msg: str) -> str:
        return "[{}] {}".format(self.name, msg)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(self.format(msg), *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.info(self.format(msg), *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.warning(self.format(msg), *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.error(self.format(msg), *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.exception(self.format(msg), *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.critical(self.format(msg), *args, **kwargs)

    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.log(level, self.format(msg), *args, **kwargs)

    def isEnabledFor(self, level: int) -> bool:
        return self.logger.isEnabledFor(level)


def get_named_client_logger(
    output_dir: str,
    name: str,
    port: int,
    host: str = 'localhost',
    filename: Optional[str] = None,
    logging_config: Optional[Dict] = None,
) -> 'PicklableClientLogger':
    logger = PicklableClientLogger(
        output_dir=output_dir,
        name=name,
        host=host,
        port=port,
        filename=filename,
        logging_config=logging_config,
    )
    return logger


def _get_named_client_logger(
    output_dir: str,
    host: str = 'localhost',
    port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
    filename: Optional[str] = None,
    logging_config: Optional[Dict] = None,
) -> logging.Logger:
    """
    When working with a logging server, clients are expected to create a logger using
    this method. For example, the automl object will create a master that awaits
    for records sent through tcp to localhost.

    Ensemble builder will then instantiate a logger object that will submit records
    via a socket handler to the server.

    We do not need to use any format as the server will render the msg as it
    needs to.

    Parameters
    ----------
        output_dir: (str)
            The location where the named log will be created
        host: (str)
            Address of where the server is gonna look for messages
        port: (str)
            The port that receives the msgs
        filename: (str)
            The filename to overwrite the default handler filename
        logging_config: (dict)
            A user specified logging configuration

    Returns
    -------
        local_loger: a logger object that has a socket handler
    """

    # local_logger is mainly used to create a TCP record which is going to be formatted
    # and handled by the main logger server. The main logger server sets up the logging format
    # that is desired, so we just make sure of two things.
    # First the local_logger below should not have extra handlers, or else we will be unecessarily
    # dumping more messages, in addition to the Socket handler we create below
    # Second, during each multiprocessing spawn, a logger is created
    # via the logger __setstate__, which is expensive. This is better handled with using
    # the multiprocessing logger
    local_logger = multiprocessing.get_logger()

    # Under this perspective, we print every msg (DEBUG) and let the server decide what to
    # dump. Also, the no propagate disable the root setup to interact with the client
    local_logger.propagate = False
    local_logger.setLevel(logging.DEBUG)

    # We also need to make sure that we only have a single socket handler.
    # The logger is a singleton, but the logger.handlers is a list. So we need to
    # check if it already has a socket handler on it
    socketHandler = logging.handlers.SocketHandler(host, port)
    if not any([handler for handler in local_logger.handlers if 'SocketHandler' in str(handler)]):
        local_logger.addHandler(socketHandler)

    return local_logger


class LogRecordStreamHandler(socketserver.StreamRequestHandler):
    """Handler for a streaming logging request.

    This basically logs the record using whatever logging policy is
    configured locally.
    """

    def handle(self) -> None:
        """
        Handle multiple requests - each expected to be a 4-byte length,
        followed by the LogRecord in pickle format. Logs the record
        according to whatever policy is configured locally.
        """
        while True:
            chunk = self.connection.recv(4)  # type: ignore[attr-defined]
            if len(chunk) < 4:
                break
            slen = struct.unpack('>L', chunk)[0]
            chunk = self.connection.recv(slen)  # type: ignore[attr-defined]
            while len(chunk) < slen:
                chunk = chunk + self.connection.recv(slen - len(chunk))  # type: ignore[attr-defined]  # noqa: E501
            obj = self.unPickle(chunk)
            record = logging.makeLogRecord(obj)
            self.handleLogRecord(record)

    def unPickle(self, data: Any) -> Any:
        return pickle.loads(data)

    def handleLogRecord(self, record: logging.LogRecord) -> None:
        # logname is define in LogRecordSocketReceiver
        # Yet Mypy Cannot see this. This is needed so that we can
        # re-use the logging setup for autosklearn into the recieved
        # records
        if self.server.logname is not None:  # type: ignore  # noqa
            name = self.server.logname  # type: ignore  # noqa
        else:
            name = record.name
        logger = logging.getLogger(name)
        # N.B. EVERY record gets logged. This is because Logger.handle
        # is normally called AFTER logger-level filtering. If you want
        # to do filtering, do it at the client end to save wasting
        # cycles and network bandwidth!
        logger.handle(record)


def start_log_server(
    host: str,
    logname: str,
    event: threading.Event,
    port: multiprocessing.Value,
    output_dir: str,
    filename: str,
    logging_config: Dict,
) -> None:
    setup_logger(
        output_dir=output_dir,
        filename=filename,
        logging_config=logging_config,
    )

    while True:
        # Loop until we find a valid port
        _port = random.randint(10000, 65535)
        try:
            receiver = LogRecordSocketReceiver(
                host=host,
                port=_port,
                logname=logname,
                event=event,
            )
            with port.get_lock():
                port.value = _port
            receiver.serve_until_stopped()
            break
        except OSError:
            continue


class LogRecordSocketReceiver(socketserver.ThreadingTCPServer):
    """
    This class implement a entity that receives tcp messages on a given address
    For further information, please check
    https://docs.python.org/3/howto/logging-cookbook.html#configuration-server-example
    """

    allow_reuse_address = True

    def __init__(
        self,
        host: str = 'localhost',
        port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
        handler: Type[LogRecordStreamHandler] = LogRecordStreamHandler,
        logname: Optional[str] = None,
        event: threading.Event = None,
    ):
        socketserver.ThreadingTCPServer.__init__(self, (host, port), handler)
        self.timeout = 1
        self.logname = logname
        self.event = event

    def serve_until_stopped(self) -> None:
        while True:
            rd, wr, ex = select.select([self.socket.fileno()],
                                       [], [],
                                       self.timeout)
            if rd:
                self.handle_request()
            if self.event is not None and self.event.is_set():
                break
