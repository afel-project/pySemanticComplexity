# -*- coding: utf-8 -*-
"""Batch utility to retrieve types of concepts from a concepts json file and save them into a single json file."""
import glob
import logging
import os
import ujson as json
from argparse import Namespace, ArgumentParser
from typing import Optional, Set

from sklearn.externals.joblib import Parallel, delayed

import batchprocessing.semantic.conceptsEnrichment as ConceptsEnrichment
from parsers.semantic.dbpediaClients import LinksCountEntitiesRetriever, EntitiesTypesRetriever
from parsers.semantic.model import TextConcepts
from utils.commons import BatchProcess

LOG = logging.getLogger(__name__)

__all__ = ['Concepts2Info']


class Concepts2Info(BatchProcess):

    def _configure_args(self, parser: ArgumentParser) -> ArgumentParser:
        parser.description = "A parallel implementation of batch processing for DBpedia concepts to concepts " \
                             "information."
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

    @classmethod
    def _load_concepts(cls, dir_in: str, ext_in: str, num_cores: int, backend: str = 'threading') -> Set[str]:
        file_names = (file_in for file_in in glob.glob(os.path.join(dir_in, '*' + ext_in)))
        concepts_sets = Parallel(n_jobs=num_cores, verbose=5, backend=backend)(
            delayed(cls._load_concept)(f) for f in file_names
        )
        return set.union(*concepts_sets)

    @classmethod
    def _load_concept(cls, filename) -> Set[str]:
        with open(filename, 'r') as f_in:
            concepts = TextConcepts.from_dict(json.load(f_in)).concepts
        return set(concept.uri for concept in concepts)

    def _run(self, args: Namespace) -> Optional[int]:
        # do a little try to avoid write error at the end
        f = open(args.out_file, 'w')
        f.close()
        del f

        LOG.info("Load concepts...")
        concepts_uri = self._load_concepts(args.data_in_dir, '.json', args.num_cores)
        LOG.info("Retrieve concepts info...")
        types_retriever = EntitiesTypesRetriever(sparql_endpoint=args.sparql_ep,
                                                 max_entities_per_query=args.max_concepts,
                                                 nice_to_server=args.nice_server)
        links_retriever = LinksCountEntitiesRetriever(sparql_endpoint=args.sparql_ep,
                                                      max_entities_per_query=50,
                                                      nice_to_server=args.nice_server)
        concepts_infos = ConceptsEnrichment.get_concepts_uris_information(concepts_uri, types_retriever,
                                                                          links_retriever)
        LOG.info("Save info...")
        with open(args.out_file, 'w') as f:
            json.dump(concepts_infos, f)
        LOG.info("Work done.")
        return


if __name__ == '__main__':
    Concepts2Info().start()
