# -*- coding: utf-8 -*-
"""Batch utility to convert graphs json files into a single csv file"""
import glob
import logging
import os
from abc import ABCMeta
from argparse import Namespace, ArgumentParser
from typing import Optional, Set

import pandas as pd
from sklearn.externals.joblib import Parallel, delayed

from dbpediaProcessing.graphs import GraphBuilderFactory, NetworkXGraphBuilder
from dbpediaProcessing.graphsTransformers import NamespaceNetworkxGraphTransformer, NetworkxGraphTransformer
from utils.commons import BatchProcess, AVAILABLE_ONTOLOGIES

LOG = logging.getLogger(__name__)

__all__ = ['GraphsToSemanticVectors', 'GraphsToSemanticVectorsRunner']


class GraphsToSemanticVectorsRunner(metaclass=ABCMeta):

    @classmethod
    def graphs_to_csv(cls, dir_in: str, csv_filename: str, ext_in: str, num_cores: int,
                      ontology_keys: Set[str], backend: str = 'threading'):
        LOG.info("Creating transformer")
        if not ontology_keys:
            managed_ns = dict((key, uri) for key, uri, _, _ in AVAILABLE_ONTOLOGIES)
        else:
            managed_ns = dict((key, uri) for key, uri, _, _ in AVAILABLE_ONTOLOGIES if key in ontology_keys)

        transformer = NamespaceNetworkxGraphTransformer(managed_ns)

        file_names = (file_in for file_in in glob.glob(os.path.join(dir_in, '*' + ext_in)))

        LOG.info("Creating vectors")
        vectors = Parallel(n_jobs=num_cores, verbose=5, backend=backend)(
            delayed(cls._graph_to_vector)(f, transformer) for f in file_names
        )

        LOG.info("Creating dataset")
        dataset = pd.DataFrame(vectors, columns=transformer.get_features_names())
        LOG.info("Adding filenames to dataset")
        dataset.insert(0, 'filename', [os.path.splitext(os.path.basename(file_in))[0]
                                       for file_in in glob.glob(os.path.join(dir_in, '*' + ext_in))])

        LOG.info("Writing dataset")
        dataset.to_csv(csv_filename, index=False)

    @classmethod
    def _graph_to_vector(cls, file_in: str, transformer: NetworkxGraphTransformer):
        graph = NetworkXGraphBuilder.from_json(file_in)
        graph_vector = transformer.vectorize_graph(graph)
        return graph_vector


class GraphsToSemanticVectors(BatchProcess):

    def _configure_args(self, parser: ArgumentParser) -> ArgumentParser:
        parser.description = "A parallel implementation of batch processing for Concept Graphs to csv file " \
                             "of vectors translation."
        parser.add_argument('data_in_dir', help='Graphs input directory', metavar='<Graphs Directory>', type=str)
        parser.add_argument('out_file', help='output CSV file', metavar='<Output file>', type=str)
        parser.add_argument('-nc', '--num-cores', help='number of cores (default: 1)', type=int, default=1)
        parser.add_argument('-on', '--ontology', help='Ontologies to used (Default: all)', type=str, action='append',
                            choices=GraphBuilderFactory.get_available_default_ontology_keys(), default=[])
        return parser

    def _run(self, args: Namespace) -> Optional[int]:
        self._logger.info("Start working...")
        GraphsToSemanticVectorsRunner.graphs_to_csv(args.data_in_dir, args.out_file, '.json',
                                                    args.num_cores, args.ontology)
        self._logger.info("Work done.")
        return


if __name__ == '__main__':
    GraphsToSemanticVectors().start()
