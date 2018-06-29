# -*- coding: utf-8 -*-
"""Batch utility to retrieve types of concepts from a concepts json file and save them into a single json file."""
import glob
import logging
import os
import ujson as json
from abc import ABCMeta
from argparse import Namespace, ArgumentParser
from typing import Optional

from sklearn.externals.joblib import Parallel, delayed

from dbpedia.concept import ConceptTypeRetriever
from dbpedia.entities import DBpediaResource
from utils.commons import BatchProcess

LOG = logging.getLogger(__name__)

__all__ = ['ConceptsTypesScrapper']


class ConceptsTypesScrapperRunner(metaclass=ABCMeta):

    @classmethod
    def load_concepts(cls, dir_in: str, ext_in: str, num_cores: int, backend: str = 'threading'):
        file_names = (file_in for file_in in glob.glob(os.path.join(dir_in, '*' + ext_in)))
        concepts_sets = Parallel(n_jobs=num_cores, verbose=5, backend=backend)(
            delayed(cls._load_concept)(f) for f in file_names
        )
        return set.union(*concepts_sets)

    @classmethod
    def _load_concept(cls, filename):
        with open(filename, 'r') as f_in:
            resources = json.load(f_in)
        concepts = [DBpediaResource.from_dict(d['concept']) for d in resources]
        return set(concept.uri for concept in concepts)


class ConceptsTypesScrapper(BatchProcess):

    def _configure_args(self, parser: ArgumentParser) -> ArgumentParser:
        parser.description = "A parallel implementation of batch processing for DBpedia concepts to concepts types."
        parser.add_argument('data_in_dir', help='Concepts .json files input directory', metavar='<Concepts Directory>',
                            type=str)
        parser.add_argument('out_file', help='Json output file', metavar='<Json Output File>', type=str)
        parser.add_argument('-se', '--sparql-ep', help='SPARQL endpoint (default: "http://dbpedia.org/sparql")',
                            type=str, default='http://dbpedia.org/sparql')
        parser.add_argument('-mc', '--max-concepts', help='Max concepts per query (default: 100)',
                            type=int, default=100)
        parser.add_argument('-nc', '--num-cores', help='Number of cores (default: 1)', type=int, default=1)
        parser.add_argument('--nice-server', help='Be nice to server by waiting randomly from 0 to 1 second between '
                                                  'each request (disabled by default, but recommended)',
                            action='store_false')
        return parser

    def _run(self, args: Namespace) -> Optional[int]:
        # do a little try to avoid write error at the end
        f = open(args.out_file, 'w')
        f.close()
        del f

        LOG.info("Load concept...")
        raw_concepts = ConceptsTypesScrapperRunner.load_concepts(args.data_in_dir, '.json', args.num_cores)
        LOG.info("Retrieve types...")
        retriever = ConceptTypeRetriever(args.sparql_ep, args.max_concepts, args.nice_server)
        concepts_types = retriever.retrieve_raw_concepts_types(raw_concepts)
        LOG.info("Save types...")
        with open(args.out_file, 'w') as f:
            json.dump(concepts_types, f)
        LOG.info("Work done.")
        return


if __name__ == '__main__':
    ConceptsTypesScrapper().start()
