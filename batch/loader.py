# -*- coding: utf-8 -*-
"""A lazy class loader to load and launch subprograms."""

from collections import namedtuple
from typing import ClassVar

__all__ = ['SubProgramLoader']

SubProgram = namedtuple('SubProgram', ['name', 'description'])


class Singleton(type):
    def __init__(cls, *args, **kwargs):
        cls.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__call__(*args, **kwargs)
        return cls.__instance


class SubProgramLoader(metaclass=Singleton):
    _SUB_PROGRAMS = [
        SubProgram(name='texts2vectors',
                   description="Full pipeline: converts raw texts files to a single CSV file of complexity vectors."),
        SubProgram(name='texts2concepts',
                   description="Converts raw texts files to JSON concepts files. Use a DBPedia Spotlight REST Api."),
        SubProgram(name='concepts2types',
                   description="Retrieves all the types of the different concepts from JSON concepts files"),
        SubProgram(name='concepts2graphs',
                   description="Converts JSON concepts files into JSON graph files"),
        SubProgram(name='graphs2vectors',
                   description="Create a CSV file of semantic complexity metrics from JSON graph files"),
        SubProgram(name='printGraph',
                   description="Draw a graph from a JSON graph file"),
    ]

    def available_subprograms(self):
        return self._SUB_PROGRAMS

    def available_names(self):
        return [prg.name for prg in self._SUB_PROGRAMS]

    def load_subprogram_class(self, name) -> ClassVar:
        if name == 'texts2concepts':
            from .texts2concepts import Texts2Concepts
            return Texts2Concepts
        elif name == 'concepts2types':
            from .concepts2types import Concepts2Types
            return Concepts2Types
        elif name == 'concepts2graphs':
            from .concepts2graphs import Concept2Graphs
            return Concept2Graphs
        elif name == 'graphs2vectors':
            from .graphs2vectors import GraphsToSemanticVectors
            return GraphsToSemanticVectors
        elif name == 'printGraph':
            from .printGraph import PrintGraph
            return PrintGraph
        elif name == 'texts2vectors':
            from .text2vectors import Texts2Vectors
            return Texts2Vectors
        else:
            raise ValueError('Bad program name')
