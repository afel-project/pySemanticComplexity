# -*- coding: utf-8 -*-
"""Concept package deals with concept information retrieving from DBPedia resources."""
import gc
import logging
import time
from collections import defaultdict
from typing import Iterable, Sized, Union

import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON

from .entities import DBpediaResource

__all__ = ['ConceptTypeRetriever']

LOG = logging.getLogger(__name__)


class ConceptTypeRetriever:
    """
    A types retriever that relies on SPARQL to find the types of a DBpedia resource
    """
    _QUERY_TEMPLATE_BASE = """select ?subject ?type 
WHERE {
    FILTER(?subject IN (%s)) .
    ?subject a ?type .
    FILTER isIRI(?type) .
}
ORDER BY ?subject ?type
"""

    def __init__(self, sparql_endpoint: str, max_concepts_per_query: int = 100, nice_to_server=True):
        self.sparql_endpoint = sparql_endpoint
        self.max_concepts_per_query = max_concepts_per_query
        self.nice_to_server = nice_to_server

    def retrieve_and_enhance_resources(self, resources: [DBpediaResource]):
        concepts_types = self.retrieve_resource_types(resources)
        for rsc in resources:
            rsc.types = concepts_types.get(rsc.uri, rsc.types)
        return resources

    def retrieve_resource_types(self, resources: [DBpediaResource]):
        raw_concepts = set(rsc.uri for rsc in resources)
        return self.retrieve_raw_concepts_types(raw_concepts)

    def retrieve_raw_concepts_types(self, concepts: Union[Iterable[str], Sized]):
        offset = 0
        limit = 10000
        data = None

        if len(concepts) > self.max_concepts_per_query:
            LOG.info("Too many concepts, splitting them in group of %d..." % self.max_concepts_per_query)
            if isinstance(concepts, set):
                concepts = list(concepts)
            for sub_concepts in (concepts[x:x + self.max_concepts_per_query] for x in
                                 range(0, len(concepts), self.max_concepts_per_query)):
                LOG.info("%d concepts will be queried..." % len(sub_concepts))
                data = self._concat_concepts_types(data, self.retrieve_raw_concepts_types(sub_concepts))

                gc.collect()
                if self.nice_to_server:
                    t = np.random.randint(1000) / 1000
                    time.sleep(t)
        else:
            while True:
                LOG.debug("%d concepts are about to be queried..." % len(concepts))
                q = self._create_query(concepts, offset=offset, limit=limit)
                sub_data = self._execute_query(q)
                data = self._concat_concepts_types(data, sub_data)

                if self.nb_types(sub_data) == limit:
                    offset += limit
                    LOG.debug("Maximum result obtained in one request. Keep interrogating...")
                else:
                    break
        return data

    @staticmethod
    def nb_types(concepts_types: dict):
        return np.sum([len(types) for types in concepts_types.values()])

    @classmethod
    def _create_query(cls, concepts: Union[Iterable[str], Sized], offset: int = 0, limit: int = 20000):
        query = cls._QUERY_TEMPLATE_BASE % ", ".join('<' + uri + '>' for uri in concepts)
        if offset is not None:
            query += ("\nOFFSET %d" % offset)
        if limit is not None:
            query += ("\nLIMIT %d" % limit)

        return query

    @staticmethod
    def _create_concepts_types():
        return defaultdict(lambda: set())

    @staticmethod
    def _concat_concepts_types(ct_1: dict, ct_2: dict):
        if ct_1 is None:
            return ct_2
        if ct_2 is None:
            return ct_1
        for subject, types in ct_2.items():
            ct_1[subject].update(types)
        return ct_1

    def _execute_query(self, query: str):
        sparql = SPARQLWrapper(self.sparql_endpoint)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        res = sparql.query()
        if 'x-sparql-maxrows' in res.info():
            raise ValueError("Too many rows in SPARQL result!")

        data = None
        try:
            info = self._create_concepts_types()
            data = res.convert()
            for triple in data['results']['bindings']:
                info[triple['subject']['value']].add(triple['type']['value'])
            return info
        finally:
            del data, res
