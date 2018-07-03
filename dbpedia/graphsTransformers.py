# -*- coding: utf-8 -*-
import logging
from itertools import chain, combinations
from typing import Dict, Union

import networkx as nx
import numpy as np
from rdflib import Namespace
from sklearn.base import BaseEstimator

from dbpedia.graphs import OntologyManager

LOG = logging.getLogger(__name__)

__all__ = ['NetworkxGraphTransformer', 'NamespaceNetworkxGraphTransformer']


class NetworkxGraphTransformer(BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit_transform(self, X, y=None):
        return np.vectorize(self.vectorize_graph, signature='(n)->(n,m)')(X)

    def vectorize_graph(self, graph: nx.Graph) -> [float]:
        diameter = self.feat_diameter(graph)
        return np.hstack([
            self.feat_nb_concepts(graph),
            self.feat_nb_unique_concepts(graph),
            self.feat_nb_nodes(graph),
            self.feat_radius(graph),
            diameter,
            self.feat_assortativity(graph),
            self.feat_density(graph),
            self.feat_text_dentity(graph, diameter)
        ]).astype(float)

    def get_features_names(self) -> [str]:
        return ["nbConcepts", "nbUniqueConcepts", "nbNodes", "radius", "diameter", "assortativity", "density",
                "textDensityMean", 'textDensityStd']

    @staticmethod
    def feat_nb_concepts(graph: nx.Graph):
        return sum((v['count'] for n, v in graph.nodes.items() if v.get('resource') is True))

    @staticmethod
    def feat_nb_unique_concepts(graph: nx.Graph):
        return sum((1 for _, v in graph.nodes.items() if v.get('resource') is True))

    @staticmethod
    def feat_nb_nodes(graph: nx.Graph):
        return len(graph.nodes)

    @staticmethod
    def feat_radius(graph: nx.Graph):
        return nx.radius(graph)

    @staticmethod
    def feat_diameter(graph: nx.Graph):
        return nx.diameter(graph)

    @staticmethod
    def feat_assortativity(graph: nx.Graph):
        return nx.degree_pearson_correlation_coefficient(graph)

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

        # TODO: Correct that!
        max_off = max(attrs.get('offset') for _, attrs in graph.nodes.items() if attrs.get('resource') is True)

        # for each combinaison of resource node, compute the text_dentisty
        resources_nodes = (node for node, attrs in graph.nodes.items() if attrs.get('resource') is True)
        comb_nodes = list(combinations(resources_nodes, r=2))

        dists_graph = np.array([nx.shortest_path_length(graph, m, n) for m, n in comb_nodes]) / diameter
        dists_text = np.abs(np.array([graph.nodes[n]['offset'] - graph.nodes[m]['offset']
                                      for m, n in comb_nodes])) / max_off

        densities = dists_text * dists_text * np.sqrt(dists_graph)
        del comb_nodes, resources_nodes, dists_text, dists_graph

        # return the mean and the standard deviation of the texual densities
        return [np.mean(densities), np.std(densities)]


class NamespaceNetworkxGraphTransformer(NetworkxGraphTransformer):
    def __init__(self, managed_namespaces: Dict[str, Union[Namespace, str]]):
        super().__init__()
        self.managed_namespaces = managed_namespaces

    def vectorize_graph(self, graph: nx.Graph) -> [float]:
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
