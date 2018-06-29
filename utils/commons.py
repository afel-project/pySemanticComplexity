# -*- coding: utf-8 -*-
"""Common utility classes, functions or constants used in the whole program."""
import logging
import logging.config
import os
import sys
from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Optional

__all__ = ['BatchProcess', 'VENDOR_DIR_PATH', 'file_can_be_write']

VENDOR_DIR_PATH = './vendor'


def file_can_be_write(filename: str) -> bool:
    return os.access(filename, os.W_OK) or \
           (not os.path.exists(filename) and os.access(os.path.split(filename)[0], os.W_OK))


class BatchProcess(metaclass=ABCMeta):
    """
    Abstract class to model a batch process program.
    """

    def __init__(self, use_logger: bool = True, progname: str = None):
        self._parser: ArgumentParser = None
        self._use_logger: bool = use_logger
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

    def _configure_logging(self, configuration_file: str = None, level: int=logging.INFO):
        """
        Configure le logging
        :param configuration_file: a configuration file path
        :param level: a global loggin level
        :return: the root loggger of the program
        """
        if configuration_file is not None:
            logging.config.fileConfig(configuration_file)
        else:
            log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
            root_logger = logging.getLogger()
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(log_formatter)
            root_logger.addHandler(console_handler)
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
            self._parser.add_argument('--logging-file', help='Logging configuration file.', type=str)

        args = self._parser.parse_args()

        if self._use_logger:
            self._configure_logging(args.logging_file)

        ret = self._run(args)

        if ret is None:
            sys.exit(0)
        else:
            sys.exit(ret)
