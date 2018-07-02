# -*- coding: utf-8 -*-
from itertools import chain
from typing import Dict, Union

import networkx as nx
import numpy as np
from rdflib import Namespace
from sklearn.base import BaseEstimator

from dbpedia.graphs import OntologyManager

__all__ = ['NetworkxGraphTransformer', 'NamespaceNetworkxGraphTransformer']


class NetworkxGraphTransformer(BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit_transform(self, X, y=None):
        return np.vectorize(self.vectorize_graph, signature='(n)->(n,m)')(X)

    def vectorize_graph(self, graph: nx.Graph) -> [float]:
        return np.array([
            self.feat_nb_concepts(graph),
            self.feat_nb_unique_concepts(graph),
            self.feat_nb_nodes(graph),
            self.feat_radius(graph),
            self.feat_diameter(graph),
            self.feat_assortativity(graph),
            self.feat_density(graph)
        ], dtype=np.float)

    def get_features_names(self) -> [str]:
        return ["nbConcepts", "nbUniqueConcepts", "nbNodes", "radius", "diameter", "assortativity", "density"]

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
        filtered_nodes = list(n for n in graph.nodes if graph[n].get('resource', False) is True
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
