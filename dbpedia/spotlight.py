# -*- coding: utf-8 -*-
"""Provides client to DBPedia Spotlight API."""
import logging
from typing import List, Tuple, Optional

import requests as rqst

from .entities import AnnotationScore, DBpediaResource

LOG = logging.getLogger(__name__)

__all__ = ['DBpediaSpotlightClient']


class DBpediaSpotlightClient:
    _DFLT_RQ_HEADERS = {
        'Accept': 'application/json'
    }

    def __init__(self, endpoint: str):
        self._endpoint = endpoint

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def annotate(self, text, confidence: float = 0.5, support: float = None, types: str = None, sparql: str = None,
                 policy: str = None) -> List[Tuple[DBpediaResource, AnnotationScore]]:
        """
        Send a request to a to annotate a text, and return an array of annotation
        :param text: (str) text to be annotated.
        :param confidence: (number) a confidence score for disambiguation. Default to 0.5.
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
            ('text', text), ('confidence', confidence), ('support', support), ('types', types),
            ('sparql', sparql), ('policy', policy))))
        rq = rqst.get(query_url, params=params, headers=self._DFLT_RQ_HEADERS)
        rq.raise_for_status()
        raw_res = rq.json()
        return list(filter(lambda x: x is not None, (self._build_resource_annotation(item)
                                                     for item in raw_res.get('Resources', []))))

    @staticmethod
    def _build_resource_annotation(json_item: dict) -> Optional[Tuple[DBpediaResource, AnnotationScore]]:
        if '@URI' not in json_item:
            return None

        types = json_item.get('@types', "")
        types = types.split(',') if types else []

        resource = DBpediaResource(uri=json_item['@URI'],
                                   types=types)

        annotation_score = AnnotationScore(offset=int(json_item['@offset']) if '@offset' in json_item else None,
                                           surface_form=json_item.get('@surfaceForm'),
                                           support=int(json_item['@support']) if '@support' in json_item else None,
                                           similarity_score=float(json_item['@similarityScore'])
                                           if '@similarityScore' in json_item else None,
                                           percentage_second_rank=float(
                                               json_item['@percentageOfSecondRank']
                                               if '@percentageOfSecondRank' in json_item else None))
        return resource, annotation_score
