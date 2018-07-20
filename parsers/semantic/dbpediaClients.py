# -*- coding: utf-8 -*-
"""ConceptExtractors package deals with concept information retrieving from DBPedia resources."""
import gc
import logging
import time
from collections import defaultdict, namedtuple
from typing import Iterable, Sized, Union, Dict, List, Set, Tuple, Any, Optional, Generator

import numpy as np
import requests as rqst
from SPARQLWrapper import SPARQLWrapper, JSON
from SPARQLWrapper.SPARQLExceptions import EndPointInternalError

from .model import DBpediaResource, AnnotationScore

__all__ = ['EntitiesTypesRetriever', 'DBpediaSpotlightClient', 'EntityCount', 'LinksCountEntitiesRetriever']

LOG = logging.getLogger(__name__)


class DBpediaSpotlightClient:
    _DFLT_RQ_HEADERS = {
        'Accept': 'application/json'
    }

    def __init__(self, endpoint: str, confidence: float = 0.5):
        self._endpoint = endpoint
        self.confidence = confidence

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def annotate(self, text, support: float = None, types: str = None, sparql: str = None,
                 policy: str = None) -> List[DBpediaResource]:
        """
        Send a request to a to annotate a text, and return an array of annotation
        :param text: (str) text to be annotated.
        :param support: (number) how prominent is this entity in Lucene Model, i.e. number of inlinks in Wikipedia.
        Default to None.
        :param types: (str) types filter (Eg.DBpedia:Place). Default to None.
        :param sparql: (str) SPARQL filtering. Default to None.
        :param policy: (str) (whitelist) select all entities that have the same type;  (blacklist) - select all
        entities that have not the same type. Default to None.
        :return: a list of couples (DBPediaResource, AnnotationScore)
        """
        query_url = self._endpoint
        LOG.debug("Annotating '%s'..." % text)
        params = dict(filter(lambda x: x[1] is not None, (
            ('text', text), ('confidence', self.confidence), ('support', support), ('types', types),
            ('sparql', sparql), ('policy', policy))))
        rq = rqst.get(query_url, params=params, headers=self._DFLT_RQ_HEADERS)
        rq.raise_for_status()
        raw_res = rq.json()
        return list(filter(lambda x: x is not None, (self._build_resource_annotation(item)
                                                     for item in raw_res.get('Resources', []))))

    @staticmethod
    def _build_resource_annotation(json_item: dict) -> Optional[DBpediaResource]:
        if '@URI' not in json_item:
            return None

        types = json_item.get('@types', "")
        types = types.split(',') if types else []

        annotation_score = AnnotationScore(offset=int(json_item['@offset']) if '@offset' in json_item else None,
                                           surface_form=json_item.get('@surfaceForm'),
                                           support=int(json_item['@support']) if '@support' in json_item else None,
                                           similarity_score=float(json_item['@similarityScore'])
                                           if '@similarityScore' in json_item else None,
                                           percentage_second_rank=float(
                                               json_item['@percentageOfSecondRank']
                                               if '@percentageOfSecondRank' in json_item else None))
        resource = DBpediaResource(uri=json_item['@URI'], types=types, scores=annotation_score)

        return resource


class EntitiesTypesRetriever:
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

    def __init__(self, sparql_endpoint: str, max_entities_per_query: int = 100, nice_to_server=True):
        self.sparql_endpoint = sparql_endpoint
        self.max_entities_per_query = max_entities_per_query
        self.nice_to_server = nice_to_server
        self.max_rows_header_name = 'x-sparql-maxrows'

    def retrieve_types_from_entities_iris(self, resources_iris: Union[Iterable[str], Sized]) -> Dict[str, Set[str]]:
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
        for rsc_subset in self._set_to_sublist_generator(resources_iris, self.max_entities_per_query):
            data = self._concat_entities_types(data, self._retrieve_types_from_sub_rsc(rsc_subset))
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
            data = self._concat_entities_types(data, res_data)
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
    def _concat_entities_types(ct_1: Dict[str, Set], ct_2: Dict[str, Set]) -> Dict[str, Set]:
        if ct_1 is None:
            return ct_2
        if ct_2 is None:
            return ct_1
        for subject, types in ct_2.items():
            ct_1[subject].update(types)
        return ct_1


EntityCount = namedtuple('EntityCount', ['inlinks', 'outlinks'])


class LinksCountEntitiesRetriever:
    _QUERY_LINK_IN_TEMPLATE_BASE = """select ?entity (COUNT(?subject) AS ?nLinks)
WHERE {
?subject ?p1 ?entity .
FILTER(?entity IN (%s)) .
}
GROUP BY ?entity"""
    _QUERY_LINK_OUT_TEMPLATE_BASE = """select ?entity (COUNT(?object) AS ?nLinks)
WHERE {
?entity ?p2 ?object .
FILTER(?entity IN (%s)) .
}
GROUP BY ?entity"""

    def __init__(self, sparql_endpoint: str, max_entities_per_query: int = 100, nice_to_server: bool = False):
        self.sparql_endpoint = sparql_endpoint
        self.nice_to_server = nice_to_server
        self.max_entities_per_query = max_entities_per_query

    def retrieve_entities_links_count(self, entities_iris: Union[Iterable[str], Sized]) -> Dict[str, EntityCount]:
        if not hasattr(entities_iris, '__getitem__'):
            entities_iris = list(entities_iris)
        sub_entities_iris_gen = (entities_iris[i:i + self.max_entities_per_query] for i in
                                 range(0, len(entities_iris), self.max_entities_per_query))
        links_count = dict()
        i = 0
        for sub_entities_iris in sub_entities_iris_gen:
            sub_links_count = self._retrieve_sub_entities_links_count(sub_entities_iris)
            links_count.update(sub_links_count)
            if self.nice_to_server is True or self.nice_to_server == i:
                t = np.random.randint(1000) / 1000
                time.sleep(t)
                i = 0
            else:
                i += 1
            LOG.debug("%d concepts have been queried..." % len(sub_entities_iris))
        assert all(links_count.get(iri) is not None for iri in entities_iris)
        return links_count

    def _retrieve_sub_entities_links_count(self, entities_iris: Union[Iterable[str], Sized]) -> Dict[str, EntityCount]:
        try:
            in_query = self._create_in_query(entities_iris)
            links_in_dict = self._execute_query(in_query)
            out_query = self._create_out_query(entities_iris)
            links_out_dcit = self._execute_query(out_query)
            links_count = dict()
            for iri in entities_iris:
                links_count[iri] = EntityCount(inlinks=links_in_dict.get(iri, 0), outlinks=links_out_dcit.get(iri, 0))
            return links_count
        except EndPointInternalError:
            if len(entities_iris) > 1:
                LOG.warning("EndPointInternalError received for multiple iris. Trying to request them one by one...")
                links_count = dict()
                for iri in entities_iris:
                    links_count.update(self._retrieve_sub_entities_links_count([iri]))
                return links_count
            else:
                entity = list(entities_iris)[0]
                LOG.warning("Timeout received for single iri %s. Set it to 0." % entity)
                return {entity: EntityCount(0, 0)}

    def _execute_query(self, query: str) -> Dict[str, int]:
        sparql = SPARQLWrapper(self.sparql_endpoint)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        query_res = sparql.query()
        # Assert max row indicator is absent
        assert query_res.info().get('x-sparql-maxrows') is None
        # Convert and parse results
        data = query_res.convert()
        res = dict()
        for line in data['results']['bindings']:
            iri = line['entity']['value']
            nb_links = line['nLinks']['value']
            res[iri] = nb_links
        return res

    @classmethod
    def _create_in_query(cls, entities_iris: Iterable[str]) -> str:
        return cls._QUERY_LINK_IN_TEMPLATE_BASE % ", ".join(("<%s>" % iri for iri in entities_iris))

    @classmethod
    def _create_out_query(cls, entities_iris: Iterable[str]) -> str:
        return cls._QUERY_LINK_OUT_TEMPLATE_BASE % ", ".join(("<%s>" % iri for iri in entities_iris))
