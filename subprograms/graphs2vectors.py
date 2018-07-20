# -*- coding: utf-8 -*-
"""Batch utility to convert graphs json files into a single csv file"""
import logging
from argparse import Namespace, ArgumentParser
from typing import Optional, Set

import batchprocessing.semantic.graphVectorization as GraphVectorization
from parsers.semantic.graphs.builders import NetworkXGraphBuilder
from parsers.semantic.graphs.tranformers import NamespaceNetworkxGraphTransformer
from utils.commons import BatchProcess
from utils.resources import DefaultOntologies

LOG = logging.getLogger(__name__)

__all__ = ['GraphsToSemanticVectors']


class GraphsToSemanticVectors(BatchProcess):

    def _configure_args(self, parser: ArgumentParser) -> ArgumentParser:
        parser.description = "A parallel implementation of batch processing for Concept Graphs to csv file " \
                             "of vectors translation."
        parser.add_argument('data_in_dir', help='Graphs input directory', metavar='<Graphs Directory>', type=str)
        parser.add_argument('out_file', help='output CSV file', metavar='<Output file>', type=str)
        parser.add_argument('-nc', '--num-cores', help='number of cores (default: 1)', type=int, default=1)
        parser.add_argument('-on', '--ontology', help='Ontologies to used (Default: all)', type=str, action='append',
                            choices=DefaultOntologies.available_ontologies().keys(), default=[])
        return parser

    def _run(self, args: Namespace) -> Optional[int]:
        self._logger.info("Start working...")
        self._logger.info("Create ")
        self._graphs_to_csv(args.data_in_dir, args.out_file, '.json', args.num_cores, args.ontology)
        self._logger.info("Work done.")
        return

    @classmethod
    def _graphs_to_csv(cls, dir_in: str, csv_filename: str, ext_in: str, num_cores: int,
                       ontology_keys: Set[str], backend: str = 'multiprocessing'):

        LOG.info("Creating transformer")
        if not ontology_keys:
            managed_ns = DefaultOntologies.available_ontologies()
        else:
            managed_ns = dict(
                (key, uri) for key, uri in DefaultOntologies.available_ontologies() if key in ontology_keys)

        transformer = NamespaceNetworkxGraphTransformer(managed_ns)
        graph_reader = NetworkXGraphBuilder.from_json

        LOG.info("Creating datasets")
        dataset = GraphVectorization.compute_vectors_from_dir(dir_in, graph_reader, transformer, num_cores,
                                                              ext_in, backend, to_dataset=True)

        LOG.info("Writing dataset")
        dataset.to_csv(csv_filename, index=False)


if __name__ == '__main__':
    GraphsToSemanticVectors().start()
