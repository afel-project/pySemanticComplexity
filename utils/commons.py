# -*- coding: utf-8 -*-
"""Common utility classes, functions or constants used in the whole program."""
import logging
import logging.config
import os
import platform
import sys
import traceback
from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Optional

LOG = logging.getLogger(__name__)

__all__ = ['BatchProcess', 'file_can_be_write',
           'ModuleShutUpWarning', 'safe_concurrency_backend']


def safe_concurrency_backend(backend: str, urllib_used: bool = False, heavy_sharing: bool = False):
    if not backend or backend not in ['multiprocessing', 'threading']:
        LOG.warning("Unknown concurrency backend. Backend set to multiprocessing")
        backend = 'multiprocessing'
    if urllib_used is True and os.name == 'posix' and platform.system().lower() == 'darwin' and backend != 'threading':
        # Force backend to multithreading if platform is mac (know bug in liburl3)
        LOG.warning("Running on Mac, concurrency backend switched to threading for safety reason")
        backend = 'threading'
    if heavy_sharing is True and backend != 'threading':
        LOG.warning("Heavy sharing of parameters in parallel computing: using threads over processes")
        backend = 'threading'
    return backend


def file_can_be_write(filename: str) -> bool:
    return os.access(filename, os.W_OK) or \
           (not os.path.exists(filename) and os.access(os.path.split(filename)[0], os.W_OK))


class BatchProcess(metaclass=ABCMeta):
    """
    Abstract class to model a batch process program.
    """

    def __init__(self, use_logger: bool = True, progname: str = None):
        self._parser = None
        self._use_logger = use_logger
        self._logger = None
        self._progname = progname
        pass

    @abstractmethod
    def _configure_args(self, parser: ArgumentParser) -> ArgumentParser:
        """
        Configure the argument parser of the program to parse the command line args, provide help message....
        :return:
        """
        pass

    @abstractmethod
    def _run(self, args: Namespace) -> Optional[int]:
        """
        Run the program.
        :param args: The argparse namespace
        :return: exit code or None
        """
        pass

    def _configure_logging(self, configuration_file: str = None, level: int = logging.INFO, debug: bool = False):
        """
        Configure le logging
        :param configuration_file: a configuration file path
        :param level: a global loggin level
        :return: the root loggger of the program
        """
        if configuration_file is not None:
            logging.config.fileConfig(configuration_file)
        else:
            log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
            root_logger = logging.getLogger()
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(log_formatter)
            root_logger.addHandler(console_handler)
            if debug:
                root_logger.setLevel(logging.DEBUG)
            else:
                root_logger.setLevel(level)
            self._logger = root_logger
            return root_logger

    def start(self, parser: ArgumentParser = None):
        """
        Start the program. This method should not be called manually.
        :return: None
        """
        self._parser = ArgumentParser(description="Base program") if parser is None else parser
        if self._progname:
            self._parser.prog = self._progname

        # self._parser.add_argument('module', help='subprogram name.', type=str)
        self._configure_args(self._parser)

        if self._use_logger:
            self._parser.add_argument('--logging-file', help='Logging configuration file', type=str)
            self._parser.add_argument('--debug', help='Debug mode', action='store_true')

        args = self._parser.parse_args()

        if self._use_logger:
            self._configure_logging(args.logging_file, debug=args.debug)

        try:
            ret = self._run(args)
        except BaseException as e:
            ret = 1
            self._logger.debug(traceback.format_exc())
            self._logger.fatal("Fatal Error happened: %s" % str(e))

        if ret is None:
            sys.exit(0)
        else:
            sys.exit(ret)


class ModuleShutUpWarning:
    def __init__(self, module: str, shut_level: int = logging.ERROR):
        self.__rdf_logger = logging.getLogger(module)
        self.__shut_level = shut_level

    def __enter__(self):
        self.__previous_level = self.__rdf_logger.level
        self.__rdf_logger.setLevel(self.__shut_level)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__rdf_logger.setLevel(self.__previous_level)
