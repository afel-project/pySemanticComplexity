# -*- coding: utf-8 -*-
"""Batch utility to convert concepts json files to graphs json files"""
import glob
import logging
import os
import ujson as json
from abc import ABCMeta
from argparse import ArgumentParser, Namespace
from typing import Optional, Set

from sklearn.externals.joblib import Parallel, delayed

from dbpediaProcessing.entities import TextConcepts
from dbpediaProcessing.graphs import GraphBuilderFactory, NetworkXGraphBuilder
from utils.commons import BatchProcess, safe_concurrency_backend, ModuleShutUpWarning

LOG = logging.getLogger(__name__)

__all__ = ['Concept2Graphs', 'Concept2GraphsRunner']


class Concept2GraphsRunner(metaclass=ABCMeta):

    @classmethod
    def json_files_to_graphs(cls, dir_in: str, dir_out: str, ext_in: str, ext_out: str, num_cores: int, force: bool,
                             backend: str = 'multiprocessing', concepts_types_file: str = None,
                             ontology_keys: Set[str] = None):
        backend = safe_concurrency_backend(backend, heavy_sharing=True)

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

        LOG.info("Creating graph builder factory")
        factory = GraphBuilderFactory()
        LOG.info("Creating ontology manager (Can take time...)")
        factory.default_ontology_manager = factory.build_default_ontology_manager(ontology_keys)
        LOG.info("Ontology manager will used these ontologies: %s" %
                 str(factory.default_ontology_manager.get_ontology_keys()))
        LOG.info("Creating graph builder")
        graph_builder = factory.build_networkx_graph_builer(concepts_types=concepts_types)

        LOG.info("Processing concepts files")
        with ModuleShutUpWarning('rdflib'):
            Parallel(n_jobs=num_cores, verbose=5, backend=backend)(
                delayed(cls._json_file_to_graph)(f[0], f[1], graph_builder) for f in file_names
            )

    @classmethod
    def _json_file_to_graph(cls, file_in: str, file_out: str, graph_builder: NetworkXGraphBuilder):
        with open(file_in, 'r') as f_in:
            text_concepts = TextConcepts.from_dict(json.load(f_in))

        graph = graph_builder.build_graph_from_entities(text_concepts.concepts)
        graph_builder.load_text_concept_attributes(text_concepts, graph)
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
        parser.add_argument('-on', '--ontology', help='Ontologies to used (Default: all)', type=str, action='append',
                            choices=GraphBuilderFactory.get_available_default_ontology_keys(), default=[])
        return parser

    def _run(self, args: Namespace) -> Optional[int]:
        self._logger.info("Start working...")
        Concept2GraphsRunner.json_files_to_graphs(args.data_in_dir, args.data_out_dir, '.json', '.json', args.num_cores,
                                                  args.force, concepts_types_file=args.types,
                                                  ontology_keys=set(args.ontology))
        self._logger.info("Work done.")
        return


if __name__ == '__main__':
    Concept2Graphs().start()
