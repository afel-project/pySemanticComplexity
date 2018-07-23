# -*- coding: utf-8 -*-
import glob
import logging
import os
import subprocess
import tempfile
from abc import ABCMeta

from utils.resources import VENDOR_DIR_PATH

LOG = logging.getLogger(__name__)

__all__ = ['StandfordJavaProgram', 'PosTagger', 'LexParser', 'TRegexCounter', 'MemoryAllocationRule']


class Singleton(type):
    def __init__(cls, *args, **kwargs):
        cls.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__call__(*args, **kwargs)
        return cls.__instance


class MemoryAllocationRule(metaclass=Singleton):
    __slots__ = ['_tregex', '_lexparser', '_postagger']

    def __init__(self):
        self._tregex = 100
        self._lexparser = 3000
        self._postagger = 300

    @staticmethod
    def _check_mem_val(value):
        if not isinstance(value, int):
            raise TypeError('Memory value must be int')
        if value <= 0:
            raise ValueError('Memory value must be > 0')
        return value

    @property
    def tregex(self):
        return self._tregex

    @tregex.setter
    def tregex(self, value):
        self._tregex = self._check_mem_val(value)

    @property
    def lexparser(self):
        return self._lexparser

    @lexparser.setter
    def lexparser(self, value):
        self._lexparser = self._check_mem_val(value)

    @property
    def postagger(self):
        return self._postagger

    @postagger.setter
    def postagger(self, value):
        self._postagger = self._check_mem_val(value)


class StandfordJavaProgram(metaclass=ABCMeta):
    def __init__(self, classpath, main_class=None, memory=300):
        self.classpath = classpath
        self.main_class = main_class
        self.memory = memory

    @staticmethod
    def create_temporary_input_file(data):
        if not isinstance(data, bytes):
            data = str(data).encode('utf-8')
        temp_file = tempfile.NamedTemporaryFile()
        temp_file.write(data)
        temp_file.flush()
        return temp_file

    def execute_java_command_with_input(self, input_args, *args):
        command = self._build_command_args(*args)
        if isinstance(input_args, bytes):
            completed_process = subprocess.run(command, input=input_args,
                                               check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            return completed_process.stdout.decode('utf-8')
        else:
            input_args = input_args if isinstance(input_args, str) else str(input_args)
            completed_process = subprocess.run(command, input=input_args, encoding='utf-8',
                                               check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            return completed_process.stdout

    def execute_java_command_no_input(self, *args):
        command = self._build_command_args(*args)
        completed_process = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return completed_process.stdout.decode('utf-8')  # Remove last end of line

    def _build_command_args(self, *args):
        command = ["java", '-mx%dm' % self.memory, '-cp', self.classpath]
        if self.main_class:
            command.append(self.main_class)
        if args:
            command += args
        return command


class TRegexCounter(StandfordJavaProgram):
    _BASE_FOLDER = os.path.join(VENDOR_DIR_PATH, 'stanford/L2SCA-2016-06-30')

    def __init__(self):
        super().__init__(
            classpath=os.path.join(self._BASE_FOLDER, "stanford-tregex.jar"),
            main_class="edu.stanford.nlp.trees.tregex.TregexPattern",
            memory=MemoryAllocationRule().tregex)

    def count(self, pattern, text, is_file_name=False):
        if not is_file_name:
            text_file = self.create_temporary_input_file(text)
            text_file_path = text_file.name
        else:
            text_file = None
            text_file_path = text

        try:
            LOG.debug("Launch tregex java command with pattern '%s' and file path '%s'" % (pattern, text_file_path))
            result = self.execute_java_command_no_input(pattern, text_file_path, '-C', '-o')
            return int(result[:-1])  # Remove extra line jump and convert to int
        finally:
            if text_file is not None:
                text_file.close()


class LexParser(StandfordJavaProgram):
    _BASE_FOLDER = os.path.join(VENDOR_DIR_PATH, 'stanford/L2SCA-2016-06-30/stanford-parser-full-2014-01-04')

    def __init__(self):
        super().__init__(
            classpath=os.path.join(self._BASE_FOLDER, "*:"),
            main_class="edu.stanford.nlp.parser.lexparser.LexicalizedParser",
            memory=MemoryAllocationRule().lexparser)

    def parse(self, text, is_file_name=False):
        if not is_file_name:
            text_file = self.create_temporary_input_file(text)
            text_file_path = text_file.name
        else:
            text_file = None
            text_file_path = text

        try:
            result = self.execute_java_command_no_input('-outputFormat', 'penn',
                                                        'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz',
                                                        text_file_path)
            return result[:-1]  # Remove extra line jump
        finally:
            if text_file is not None:
                text_file.close()


class PosTagger(StandfordJavaProgram):
    _BASE_FOLDER = os.path.join(VENDOR_DIR_PATH, 'stanford/stanford-postagger-full-2018-02-27')
    _MODEL_EXT = '.tagger'

    def __init__(self, model='english-left3words-distsim'):
        super().__init__(
            classpath=os.path.join(self._BASE_FOLDER, "stanford-postagger.jar:"),
            main_class="edu.stanford.nlp.tagger.maxent.MaxentTagger",
            memory=MemoryAllocationRule().postagger)
        if model not in self.get_available_models():
            raise ValueError('model "%s" for POS Tagger does not exist' % model)
        self._model = model

    @classmethod
    def get_available_models(cls):
        models_directory = os.path.join(cls._BASE_FOLDER, 'models')
        return [os.path.splitext(os.path.basename(filename))[0] for filename in
                glob.glob(models_directory + '/*' + cls._MODEL_EXT)]

    @property
    def model(self):
        return self._model

    def tag_pos(self, text, is_file_name=False):
        model_path = os.path.join(os.path.join(self._BASE_FOLDER, 'models'), self._model) + self._MODEL_EXT

        if not is_file_name:
            result = self.execute_java_command_with_input(text, '-model', model_path)
        else:
            result = self.execute_java_command_no_input('-model', model_path, '-textFile', text)
        return result[:-1].split()  # Remove extra line jump and split into array
