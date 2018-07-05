# -*- coding: utf-8 -*-
from typing import List, Iterable, Set, Dict

import logging
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.externals.joblib import Parallel, delayed

from dbpediaProcessing.concept import ConceptTypeRetriever
from dbpediaProcessing.entities import TextConcepts
from dbpediaProcessing.graphs import GraphBuilder, GraphBuilderFactory
from dbpediaProcessing.graphsTransformers import GraphTransformer, NamespaceNetworkxGraphTransformer
from dbpediaProcessing.spotlight import DBpediaSpotlightClient
from textProcessing.TextTransformers import TextTransformer
from utils.commons import safe_concurrency_backend, Ontology

__all__ = ['SemanticTransformer', 'build_default_semantic_transformer']

LOG = logging.getLogger(__name__)


def build_default_semantic_transformer(spotlight_ep: str, spotlight_confidence: float, sparql_ep: str, nb_cores:int = 1):
    text_transformer = TextTransformer()
    spotlight_client = DBpediaSpotlightClient(spotlight_ep)
    concept_types_retriever = ConceptTypeRetriever(sparql_ep, 100, nice_to_server=True)
    graph_builder_factory = GraphBuilderFactory()
    if graph_builder_factory.default_ontology_manager is None:
        LOG.info("Creating default ontology manager")
        ontology_manager = graph_builder_factory.build_default_ontology_manager()
        graph_builder_factory.default_ontology_manager = ontology_manager
    graph_transformer = NamespaceNetworkxGraphTransformer(
        graph_builder_factory.default_ontology_manager.managed_namespaces)
    return SemanticTransformer(text_transformer, spotlight_client, spotlight_confidence, concept_types_retriever,
                               graph_builder_factory, graph_transformer, nb_cores=nb_cores)


class SemanticTransformer(BaseEstimator):
    def __init__(self, text_transformer: TextTransformer, spotlight_client: DBpediaSpotlightClient,
                 spotlight_confidence: float, concept_types_retriever: ConceptTypeRetriever,
                 graph_builder_factory: GraphBuilderFactory, graph_transformer: GraphTransformer, nb_cores: int = 1):
        self.text_transformer = text_transformer
        self.spotlight_client = spotlight_client
        self.spotlight_confidence = spotlight_confidence
        self.concept_types_retriever = concept_types_retriever
        self.graph_builder_factory = graph_builder_factory
        self.graph_transformer = graph_transformer
        self.nb_cores = nb_cores
        self.backend = 'multiprocessing'

    def fit_transform(self, X: Iterable[str], y=None) -> np.ndarray:
        # text to concepts list
        LOG.info("Conversion of texts to concepts lists...")
        text_concepts_list = self._texts_to_concepts_list(X)
        # Retrieve types for each concept
        LOG.info("Retrieval of types of concepts...")
        concepts_types = self._retrieve_concepts_types(text_concepts_list)
        # Compute graphs then vectors
        LOG.info("Conversion of concepts of texts to graphs and vectorization...")
        return self._compute_graphs_vectors(text_concepts_list, concepts_types)

    def get_features_names(self) -> List[str]:
        return self.graph_transformer.get_features_names()

    def _texts_to_concepts_list(self, texts: Iterable[str]) -> List[TextConcepts]:
        backend = safe_concurrency_backend(self.backend, urllib_used=True)
        return Parallel(n_jobs=self.nb_cores, verbose=5, backend=backend)(
            delayed(self._text_to_concepts)(text) for text in texts)

    def _text_to_concepts(self, text: str) -> TextConcepts:
        # Count number of words in the text
        nb_words = self.text_transformer.count_words(text)
        # Get concepts from spotlight
        concepts = self.spotlight_client.annotate(text=text, confidence=self.spotlight_confidence)
        return TextConcepts(concepts, nb_words)

    def _retrieve_concepts_types(self, text_concepts_list: List[TextConcepts]) -> Dict[str, Set[str]]:
        # Create a set of unique concepts
        concepts = set(concept for text_concepts in text_concepts_list for concept in text_concepts.concepts)
        # Retrieve all types of each concepts
        return self.concept_types_retriever.retrieve_resource_types(concepts)

    def _compute_graphs_vectors(self, text_concepts_list: List[TextConcepts], concepts_types: Dict[str, Set[str]]) \
            -> np.ndarray:
        backend = safe_concurrency_backend(self.backend, urllib_used=False)
        # Build graph builder based on concept_types
        graph_builder = self.graph_builder_factory.build_networkx_graph_builer(concepts_types=concepts_types)

        vectors = Parallel(n_jobs=self.nb_cores, verbose=5, backend=backend)(
            delayed(self._compute_graph_vectors)(text_concepts, graph_builder) for text_concepts in text_concepts_list)
        return np.array(vectors, dtype=float)

    def _compute_graph_vectors(self, text_concepts: TextConcepts, graph_builder: GraphBuilder) \
            -> Iterable[float]:
        # Create the graph of concepts, and add attributes from text_concepts
        graph = graph_builder.build_graph_from_entities(text_concepts.concepts)
        graph_builder.load_text_concept_attributes(text_concepts, graph)
        # Vectorize the graph
        return self.graph_transformer.vectorize_graph(graph)
