# -*- coding: utf-8 -*-
"""Batch utility to convert concepts json files to graphs json files"""
import glob
import logging
import os
import ujson as json
from abc import ABCMeta
from argparse import ArgumentParser, Namespace
from typing import Optional

from sklearn.externals.joblib import Parallel, delayed

from dbpedia.entities import DBpediaResource
from dbpedia.graphs import GraphBuilderFactory, NetworkXGraphBuilder
from utils.commons import BatchProcess

LOG = logging.getLogger(__name__)

__all__ = ['Concept2Graphs']


class Concept2GraphsRunner(metaclass=ABCMeta):

    @classmethod
    def json_files_to_graphs(cls, dir_in: str, dir_out: str, ext_in: str, ext_out: str, num_cores: int, force: bool,
                             backend: str = 'threading', concepts_types_file: str = None):
        file_names = ((file_in, os.path.join(dir_out, os.path.splitext(os.path.basename(file_in))[0] + ext_out))
                      for file_in in glob.glob(os.path.join(dir_in, '*' + ext_in)))
        if not force:
            file_names = filter(lambda x: not os.path.exists(x[1]), file_names)

        if concepts_types_file is not None:
            LOG.info("Using an external concepts - types mapping dictionary...")
            with open(concepts_types_file, 'r') as f:
                concepts_types = json.load(f)
        else:
            concepts_types = None

        factory = GraphBuilderFactory()
        graph_builder = factory.build_networkx_graph_builer(concepts_types=concepts_types)

        Parallel(n_jobs=num_cores, verbose=5, backend=backend)(
            delayed(cls._json_file_to_graph)(f[0], f[1], graph_builder) for f in file_names
        )

    @classmethod
    def _json_file_to_graph(cls, file_in: str, file_out: str, graph_builder: NetworkXGraphBuilder):
        with open(file_in, 'r') as f_in:
            resources = json.load(f_in)
        concepts = [DBpediaResource.from_dict(d['concept']) for d in resources]
        graph = graph_builder.build_graph_from_entities(concepts)
        graph_builder.to_json(file_out, graph)


class Concept2Graphs(BatchProcess):

    def _configure_args(self, parser: ArgumentParser) -> ArgumentParser:
        parser.description = "A parallel implementation of batch processing for DBpedia concepts to Concept Graph " \
                             "translation."
        parser.add_argument('data_in_dir', help='Concepts input directory', metavar='<Concepts Directory>', type=str)
        parser.add_argument('data_out_dir', help='Graphs output directory', metavar='<Graphs Directory>', type=str)
        parser.add_argument('-t', '--types', help='Concepts - Types external json mapping (default: None)',
                            metavar='<Concepts-Types File>', type=str, default=None)
        parser.add_argument('-f', '--force', help='Do not take care of already existing output files '
                                                  '(disabled by default)', action='store_true')
        parser.add_argument('-nc', '--num-cores', help='Number of cores (default: 1)', type=int, default=1)
        return parser

    def _run(self, args: Namespace) -> Optional[int]:
        self._logger.info("Start working...")
        Concept2GraphsRunner.json_files_to_graphs(args.data_in_dir, args.data_out_dir, '.json', '.json', args.num_cores,
                                                  args.force, concepts_types_file=args.types)
        self._logger.info("Work done.")
        return


if __name__ == '__main__':
    Concept2Graphs().start()
