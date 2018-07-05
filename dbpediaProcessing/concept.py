# -*- coding: utf-8 -*-
"""Concept package deals with concept information retrieving from DBPedia resources."""
import gc
import logging
import time
from collections import defaultdict
from typing import Iterable, Sized, Union, Dict, List, Set, Tuple, Any, Optional, Generator

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
        self.max_rows_header_name = 'x-sparql-maxrows'

    def retrieve_and_enhance_resources(self, resources: Iterable[DBpediaResource]) -> Iterable[DBpediaResource]:
        """
        For a collection of DBpedia resources, retrieve all their types, add them to the resources and return
        the enhanced resources.
        :param resources: the DBpedia resources
        :return: the DBpedia resources
        """
        concepts_types = self.retrieve_resource_types(resources)
        for rsc in resources:
            rsc.types = concepts_types.get(rsc.uri, rsc.types)
        return resources

    def retrieve_resource_types(self, resources: Iterable[DBpediaResource]) -> Dict[str, Set[str]]:
        """
        For a collection of DBpedia resources, retrieve all their types and return a map
        of resource IRI - List of types IRI.
        :param resources: the collection of DBpedia resources
        :return: a dictionnary of resource uri - List of types IRI
        """
        raw_concepts = set(rsc.uri for rsc in resources)
        return self.retrieve_types_from_resource_iris(raw_concepts)

    def retrieve_types_from_resource_iris(self, resources_iris: Union[Iterable[str], Sized]) -> Dict[str, Set[str]]:
        """
        For a sized collection of resource IRIs, retrieve all their types and return a map of
        resource IRI - List of types IRI.
        :param resources_iris: the sized collection of IRIs
        :return: a dictionnary of resource uri - List of types IRI
        """
        # Create a set of resources_iris if this is not already the case, in order to ease the process
        if not isinstance(resources_iris, set):
            resources_iris = set(resources_iris)
        # Iterate over group of max max_concepts_per_query items
        data = None
        for rsc_subset in self._set_to_sublist_generator(resources_iris, self.max_concepts_per_query):
            data = self._concat_concepts_types(data, self._retrieve_types_from_sub_rsc(rsc_subset))
            # Clean memory (can be high usage)
            gc.collect()
            # Sleep a bit if required
            if self.nice_to_server:
                t = np.random.randint(1000) / 1000
                time.sleep(t)
        return dict(data)

    def _retrieve_types_from_sub_rsc(self, rsc_subset: List[str]) -> Dict[str, Set[str]]:
        """
        For a list of resources of IRIs that respect the max_concepts_per_query, retrieve all their types.
        :param rsc_subset: the list of IRIs whose length <= self.max_concepts_per_query
        :return: a dictionnary of resource uri - List of types IRI
        """
        # Iterate while results are truncated by the server (if too many triples are given back)
        data = None
        still_triples = True
        offset_request = 0
        while still_triples:
            query = self._create_query(rsc_subset, offset=offset_request, limit=None)
            res_data, max_rows = self._execute_query(query)
            data = self._concat_concepts_types(data, res_data)
            if max_rows:
                assert len(res_data) == max_rows  # Check information consistency
                LOG.info("%d concepts have been queried, and others queries are required..." % len(rsc_subset))
            else:
                LOG.debug("%d concepts have been queried..." % len(rsc_subset))
                still_triples = False
        return data

    @staticmethod
    def nb_types(concepts_types: Dict[Any, Sized]) -> int:
        return sum([len(types) for types in concepts_types.values()])

    def _execute_query(self, query: str) -> Tuple[Dict[str, Set], int]:
        sparql = SPARQLWrapper(self.sparql_endpoint)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        query_res = sparql.query()
        # Set max row indicator if any
        max_rows = query_res.info().get(self.max_rows_header_name)
        # Convert and parse results
        data = query_res.convert()
        results = defaultdict(lambda: set())
        for triple in data['results']['bindings']:
            results[triple['subject']['value']].add(triple['type']['value'])
        return results, max_rows

    @classmethod
    def _create_query(cls, concepts: Union[Iterable[str], Sized],
                      offset: Optional[int] = 0, limit: Optional[int] = 20000) -> str:
        query = cls._QUERY_TEMPLATE_BASE % ", ".join('<' + uri + '>' for uri in concepts)
        if offset is not None:
            query += ("\nOFFSET %d" % offset)
        if limit is not None:
            query += ("\nLIMIT %d" % limit)

        return query

    @staticmethod
    def _set_to_sublist_generator(data_set: set, limit: int) -> Generator[List[str], None, None]:
        subset = []
        for data in data_set:
            subset.append(data)
            if len(subset) >= limit:
                yield subset
                subset = []
        if len(subset) > 0:
            yield subset

    @staticmethod
    def _concat_concepts_types(ct_1: Dict[str, Set], ct_2: Dict[str, Set]) -> Dict[str, Set]:
        if ct_1 is None:
            return ct_2
        if ct_2 is None:
            return ct_1
        for subject, types in ct_2.items():
            ct_1[subject].update(types)
        return ct_1
