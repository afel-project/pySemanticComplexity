# -*- coding: utf-8 -*-
"""Batch utility to convert concepts json files to graphs json files"""
import logging
import ujson as json
from argparse import ArgumentParser, Namespace
from typing import Optional, Set

import batchprocessing.semantic.graphCreation as GraphCreation
from parsers.semantic.graphs.builders import NetworkXGraphBuilder
from parsers.semantic.model import ConceptInformation
from utils.commons import BatchProcess
from utils.resources import DefaultOntologies

LOG = logging.getLogger(__name__)

__all__ = ['Concept2Graphs']


class Concept2Graphs(BatchProcess):

    def _configure_args(self, parser: ArgumentParser) -> ArgumentParser:
        parser.description = "A parallel implementation of batch processing for DBpedia concepts to Concept Graph " \
                             "translation."
        parser.add_argument('data_in_dir', help='Concepts input directory', metavar='<Concepts Directory>', type=str)
        parser.add_argument('data_out_dir', help='Graphs output directory', metavar='<Graphs Directory>', type=str)
        parser.add_argument('-t', '--info', help='Concepts - Info external json mapping (default: None)',
                            metavar='<Concepts-Info File>', type=str, default=None)
        parser.add_argument('-f', '--force', help='Do not take care of already existing output files '
                                                  '(disabled by default)', action='store_true')
        parser.add_argument('-nc', '--num-cores', help='Number of cores (default: 1)', type=int, default=1)
        parser.add_argument('-on', '--ontology', help='Ontologies to used (Default: all)', type=str, action='append',
                            choices=DefaultOntologies.available_ontologies().keys(), default=[])
        return parser

    def _run(self, args: Namespace) -> Optional[int]:
        self._logger.info("Start working...")
        self._json_files_to_graphs(args.data_in_dir, args.data_out_dir, '.json', '.json', args.num_cores,
                                   args.force, concepts_info_file=args.info,
                                   ontology_keys=set(args.ontology))
        self._logger.info("Work done.")
        return

    @classmethod
    def _json_files_to_graphs(cls, dir_in: str, dir_out: str, ext_in: str, ext_out: str, num_cores: int, force: bool,
                              backend: str = 'multiprocessing', concepts_info_file: str = None,
                              ontology_keys: Set[str] = None):
        if concepts_info_file is not None:
            LOG.info("Using an external concepts - info mapping dictionary...")
            with open(concepts_info_file, 'r') as f:
                concepts_info = ConceptInformation.load_concept_information_dict_from_json(json.load(f))
        else:
            concepts_info = None

        LOG.info("Creating ontology manager (Can take time...)")
        ontology_manager = DefaultOntologies.build_ontology_manager(ontology_keys)
        LOG.info("Create Graph builder...")
        graph_builder = NetworkXGraphBuilder(ontology_manager, concepts_info)

        LOG.info("Processing concepts files")
        GraphCreation.concepts_dir_to_graph_files(dir_in, dir_out, graph_builder, num_cores, backend, force, ext_in,
                                                  ext_out)


if __name__ == '__main__':
    Concept2Graphs().start()
