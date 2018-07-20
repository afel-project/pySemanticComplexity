# -*- coding: utf-8 -*-
import logging
from abc import ABCMeta, abstractmethod
from itertools import chain, combinations
from typing import Dict, Union, Iterable

import networkx as nx
import numpy as np
from rdflib import Namespace
from sklearn.base import BaseEstimator

from parsers.semantic.graphs.ontologies import OntologyManager

LOG = logging.getLogger(__name__)

__all__ = ['GraphTransformer', 'NetworkxGraphTransformer', 'NamespaceNetworkxGraphTransformer']


class GraphTransformer(BaseEstimator, metaclass=ABCMeta):

    def fit_transform(self, X, y=None):
        return np.vectorize(self.vectorize_graph, signature='(n)->(n,m)')(X)

    @abstractmethod
    def vectorize_graph(self, graph) -> Iterable[float]:
        pass

    @abstractmethod
    def get_features_names(self) -> [str]:
        pass


class NetworkxGraphTransformer(GraphTransformer):
    def __init__(self):
        super().__init__()

    def fit_transform(self, X, y=None):
        return np.vectorize(self.vectorize_graph, signature='(n)->(n,m)')(X)

    def vectorize_graph(self, graph: nx.Graph) -> Iterable[float]:
        diameter = self.feat_diameter(graph)
        nb_concepts = self.feat_nb_concepts(graph)
        nb_unique_concepts = self.feat_nb_unique_concepts(graph)
        nb_words = graph.graph.get('nb_words', -1)
        return np.hstack([
            nb_words,
            nb_concepts,
            nb_unique_concepts,
            nb_concepts / nb_words,
            nb_unique_concepts / nb_words,
            self.feat_nb_nodes(graph),
            self.feat_radius(graph),
            diameter,
            self.feat_assortativity(graph),
            self.feat_density(graph),
            self.feat_text_dentity(graph, diameter),
            self.feat_types_links_mean_std(graph),
        ]).astype(float)

    def get_features_names(self) -> [str]:
        return ["nbWord", "nbConcepts", "nbUniqueConcepts", "conceptsWordsRatio", "uniqueConceptsWordsRatio",
                "nbNodes", "radius", "diameter", "assortativity", "density",
                "textDensityMean", 'textDensityStd',
                "nbTypesMean", "nbTypesStd", "nbLinkInMean", "nbLinkInStd", "nbLinkOutMean", "nbLinkOutStd"]

    @staticmethod
    def feat_nb_concepts(graph: nx.Graph):
        return sum((v['count'] for n, v in graph.nodes.items() if v.get('resource') is True), 0)

    @staticmethod
    def feat_nb_unique_concepts(graph: nx.Graph):
        return sum((1 for _, v in graph.nodes.items() if v.get('resource') is True), 0)

    @staticmethod
    def feat_nb_nodes(graph: nx.Graph):
        return len(graph.nodes)

    @staticmethod
    def feat_radius(graph: nx.Graph):
        if graph:
            return nx.radius(graph)
        else:
            return 0

    @staticmethod
    def feat_diameter(graph: nx.Graph):
        if graph:
            return nx.diameter(graph)
        else:
            return 0

    @staticmethod
    def feat_assortativity(graph: nx.Graph):
        if graph:
            return nx.degree_pearson_correlation_coefficient(graph)
        else:
            return 0

    @staticmethod
    def feat_density(graph: nx.Graph):
        card_nodes = len(graph.nodes)
        return 2.0 * len(graph.edges) / (card_nodes * (card_nodes - 1)) if card_nodes > 1 else 1

    @staticmethod
    def feat_text_dentity(graph: nx.Graph, diameter: float):
        # Implementation: for n, m two different nodes, td(n,m) = dist_graph(n,m)/diameter_graph * dist_text(n,m)
        # indicator between 0 and 1.
        # O -> concept are very close in text and/or in semantic
        # 1 -> concept are very far in text and in semantic

        if not graph:
            return 0

        text_len = graph.graph.get('nb_words')
        if text_len is None:
            LOG.warning("cannot retrieve nb words from graph to compute text length")
            text_len = max(attrs.get('offset') for _, attrs in graph.nodes.items() if attrs.get('resource') is True)

        # for each combinaison of resource node, compute the text_dentisty
        resources_nodes = (node for node, attrs in graph.nodes.items() if attrs.get('resource') is True)
        comb_nodes = list(combinations(resources_nodes, r=2))

        dists_graph = np.array([nx.shortest_path_length(graph, m, n) for m, n in comb_nodes]) / diameter
        dists_text = np.abs(np.array([graph.nodes[n]['offset'] - graph.nodes[m]['offset']
                                      for m, n in comb_nodes])) / text_len

        densities = dists_text * dists_text * np.sqrt(dists_graph)
        del comb_nodes, resources_nodes, dists_text, dists_graph

        # return the mean and the standard deviation of the texual densities
        return [np.mean(densities), np.std(densities)]

    @staticmethod
    def feat_types_links_mean_std(graph: nx.Graph) -> Iterable[float]:
        """
        Compute nbTypesMean, nbTypesStd, nbLinkInMean, nbLinkInStd, nbLinkOutMean, nbLinkOutStd
        :param graph: the concepts graph
        :return: an array of 6 values
        """
        resources_nodes = ((node, attrs) for node, attrs in graph.nodes.items() if attrs.get('resource') is True)
        stats = {
            'nbTypes': [],
            'nbLinksIn': [],
            'nbLinksOut': []
        }
        for node, attrs in resources_nodes:
            for attribute, stat_list in stats.items():
                if attrs.get(attribute) is None:
                    LOG.warning("Resource node without %s (node's attributes: %s)" % (attribute, str(attrs.keys())))
                else:
                    stat_list.append(int(attrs.get(attribute)))
        return np.array([np.mean(stats['nbTypes']), np.std(stats['nbTypes']),
                         np.mean(stats['nbLinksIn']), np.std(stats['nbLinksIn']),
                         np.mean(stats['nbLinksOut']), np.std(stats['nbLinksOut'])])


class NamespaceNetworkxGraphTransformer(NetworkxGraphTransformer):
    def __init__(self, managed_namespaces: Dict[str, Union[Namespace, str]]):
        super().__init__()
        self.managed_namespaces = managed_namespaces

    def vectorize_graph(self, graph: nx.Graph) -> Iterable[float]:
        return np.hstack(chain((super().vectorize_graph(graph),),
                               (self._vectorize_partial_graph(graph, ns)
                                for ns in self.managed_namespaces.values())))

    def get_features_names(self) -> [str]:
        return super().get_features_names() + [ft for ns_key in
                                               self.managed_namespaces.keys()
                                               for ft in self._get_partial_features_names(ns_key)]

    def _vectorize_partial_graph(self, graph: nx.Graph, namespace: Union[str, Namespace]) -> [float]:
        # Filter nodes: keep only node that are resource, root, or that belong to the namespace
        filtered_nodes = list(n for n, attrs in graph.nodes.items() if attrs.get('resource') is True
                              or n == OntologyManager.get_root() or n.startswith(namespace))
        # Create a subgraph from filtered nodes
        sub_graph = graph.subgraph(filtered_nodes)
        # Compute features from subgraph
        return np.array([
            self.feat_nb_nodes(sub_graph),
            self.feat_density(sub_graph)
        ])

    @staticmethod
    def _get_partial_features_names(namespace_key: str) -> [str]:
        return [ft + '_' + namespace_key for ft in ('nbNodes', 'density')]
