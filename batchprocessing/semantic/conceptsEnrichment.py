# -*- coding: utf-8 -*-
import logging
from typing import Iterable, Dict, Set, Union, Sized

from parsers.semantic.dbpediaClients import EntitiesTypesRetriever, \
    LinksCountEntitiesRetriever, EntityCount
from parsers.semantic.model import TextConcepts, ConceptInformation

__all__ = ['get_concepts_uris', 'get_concepts_types', 'get_concepts_links_count', 'get_concepts_uris_information']

LOG = logging.getLogger(__name__)


def get_concepts_uris(texts_concepts: Iterable[TextConcepts]):
    return set(concept.uri for t_c in texts_concepts for concept in t_c.concepts)


def get_concepts_types(concepts_uris: Iterable[str], types_retriever: EntitiesTypesRetriever) \
        -> Dict[str, Set[str]]:
    return types_retriever.retrieve_types_from_entities_iris(concepts_uris)


def get_concepts_links_count(concepts_uris: Iterable[str], links_retriever: LinksCountEntitiesRetriever) \
        -> Dict[str, EntityCount]:
    return links_retriever.retrieve_entities_links_count(concepts_uris)


def get_concepts_uris_information(concepts_uris: Union[Iterable[str], Sized], types_retriever: EntitiesTypesRetriever,
                                  links_retriever: LinksCountEntitiesRetriever) -> Dict[str, ConceptInformation]:
    LOG.info("%d concepts to enrich" % len(concepts_uris))
    LOG.info("Retrieving types...")
    c_types = get_concepts_types(concepts_uris, types_retriever)
    LOG.info("Retrieving links count...")
    c_links = get_concepts_links_count(concepts_uris, links_retriever)
    LOG.info("%d links count and %d types collections found" % (len(c_links), len(c_types)))
    LOG.info("Building information dictionnary...")
    info = dict()
    for uri in concepts_uris:
        links = c_links.get(uri)
        inlinks = links.inlinks if links is not None else 0
        outlinks = links.outlinks if links is not None else 0
        info[uri] = ConceptInformation(uri, c_types.get(uri, []), inlinks, outlinks)
    return info


def get_concepts_information(texts_concepts: Iterable[TextConcepts], types_retriever: EntitiesTypesRetriever,
                             links_retriever: LinksCountEntitiesRetriever) -> Dict[str, ConceptInformation]:
    return get_concepts_uris_information(get_concepts_uris(texts_concepts), types_retriever, links_retriever)
