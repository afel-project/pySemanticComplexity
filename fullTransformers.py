# -*- coding: utf-8 -*-
import logging
from typing import List, Iterable

import numpy as np
from sklearn.base import BaseEstimator

import batchprocessing.semantic.conceptExtraction as ConceptsExtraction
import batchprocessing.semantic.conceptsEnrichment as ConceptsEnrichment
import batchprocessing.semantic.graphCreation as GraphCreation
import batchprocessing.semantic.graphVectorization as GraphVectorization
from parsers.preprocessing.text import TextPreprocessor
from parsers.semantic.dbpediaClients import DBpediaSpotlightClient, EntitiesTypesRetriever, \
    LinksCountEntitiesRetriever
from parsers.semantic.graphs.builders import NetworkXGraphBuilder
from parsers.semantic.graphs.ontologies import OntologyManager
from parsers.semantic.graphs.tranformers import NamespaceNetworkxGraphTransformer, GraphTransformer
from utils.resources import DefaultOntologies

__all__ = ['SemanticTransformer', 'build_default_semantic_transformer']

LOG = logging.getLogger(__name__)


def build_default_semantic_transformer(spotlight_ep: str, spotlight_confidence: float, sparql_ep: str,
                                       nb_cores: int = 1):
    text_preprocessor = TextPreprocessor()
    spotlight_client = DBpediaSpotlightClient(spotlight_ep, spotlight_confidence)
    entities_types_retriever = EntitiesTypesRetriever(sparql_ep, 100, nice_to_server=True)
    entities_links_retriever = LinksCountEntitiesRetriever(sparql_ep, nice_to_server=True)
    LOG.info("Creating default ontology manager")
    ontology_manager = DefaultOntologies.build_ontology_manager()
    graph_transformer = NamespaceNetworkxGraphTransformer(ontology_manager.managed_namespaces)
    return SemanticTransformer(text_preprocessor, spotlight_client, entities_types_retriever, entities_links_retriever,
                               ontology_manager, graph_transformer, nb_cores)


class SemanticTransformer(BaseEstimator):
    def __init__(self, text_preprocessor: TextPreprocessor, spotlight_client: DBpediaSpotlightClient,
                 entities_types_retriever: EntitiesTypesRetriever,
                 entities_links_retriever: LinksCountEntitiesRetriever,
                 ontology_manager: OntologyManager, graph_transformer: GraphTransformer, nb_cores: int = 1):
        self.text_preprocessor = text_preprocessor
        self.spotlight_client = spotlight_client
        self.entities_types_retriever = entities_types_retriever
        self.entities_links_retriever = entities_links_retriever
        self.ontology_manager = ontology_manager
        self.graph_transformer = graph_transformer
        self.nb_cores = nb_cores
        self.backend = 'multiprocessing'

    def fit_transform(self, X: Iterable[str], y=None) -> np.ndarray:
        # text to concepts list
        LOG.info("Conversion of texts to concepts lists...")
        text_concepts_list = ConceptsExtraction.texts_to_entities(X, self.text_preprocessor, self.spotlight_client,
                                                                  self.nb_cores, self.backend)
        # Retrieve types for each concept
        LOG.info("Retrieval of info of concepts...")
        concept_info = ConceptsEnrichment.get_concepts_information(text_concepts_list, self.entities_types_retriever,
                                                                   self.entities_links_retriever)
        # Compute graphs
        LOG.info("Concersion of concepts to graphs")
        graph_builder = NetworkXGraphBuilder(self.ontology_manager, concept_info)
        graphs = GraphCreation.compute_graphs(text_concepts_list, graph_builder, self.nb_cores, self.backend)
        del text_concepts_list, concept_info
        # Compute  vectors
        LOG.info("Vectorization of graph...")
        return np.array(GraphVectorization.compute_vectors(graphs, self.graph_transformer, self.nb_cores, self.backend))

    def get_features_names(self) -> List[str]:
        return self.graph_transformer.get_features_names()
