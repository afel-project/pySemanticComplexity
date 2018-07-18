# -*- coding: utf-8 -*-
"""Concept graph manipulation package. Provides classes to manage concepts RDF ontologies, concepts graphs and
to operate on graphs."""
import logging
import ujson as json
from abc import abstractmethod, ABCMeta
from collections import Counter
from itertools import chain
from typing import Dict

import networkx as nx
from rdflib import URIRef

from parsers.semantic.graphs.ontologies import OntologyManager
from parsers.semantic.model import TextConcepts, DBpediaResource, ConceptInformation

__all__ = ['GraphBuilder', 'NetworkXGraphBuilder']

LOG = logging.getLogger(__name__)


class GraphBuilder(metaclass=ABCMeta):
    def __init__(self, ontology_manager: OntologyManager, concepts_information: Dict[str, ConceptInformation] = None):
        self._ontology_mgr = ontology_manager
        self._concepts_information = concepts_information

    @property
    def ontology_manager(self) -> OntologyManager:
        return self._ontology_mgr

    @ontology_manager.setter
    def ontology_manager(self, value: OntologyManager):
        self._ontology_mgr = value

    @property
    def concepts_information(self) -> Dict[str, ConceptInformation]:
        return self._concepts_information

    @concepts_information.setter
    def concepts_information(self, value: Dict[str, ConceptInformation]):
        self._concepts_information = value

    @ontology_manager.deleter
    def ontology_manager(self):
        raise AttributeError("Cannot delete ontology manager.")

    def build_graph_from_text_concepts(self, text_concepts: TextConcepts):
        graph_params = dict(nb_words=text_concepts.nb_words)
        return self.build_graph_from_entities(text_concepts.concepts, graph_params)

    def build_graph_from_entities(self, resources: [DBpediaResource], graph_params: Dict = None):
        return self.build_sub_graph_from_entities(resources, graph_params, namespace_key=None)

    def build_sub_graph_from_entities(self, resources: [DBpediaResource], graph_params: Dict = None,
                                      namespace_key: str = None):
        LOG.debug("Build a graph for %d resources" % len(resources))
        if namespace_key is not None:
            # Check that the namespace key exists in the ontology manager
            assert self._ontology_mgr.get_namespace(namespace_key) is not None

        # Create a counter of resources, and a dict of uri-resource to have a unique liste of resources
        resources_counter = Counter((r.uri for r in resources))
        resources_dict = dict(((r.uri, r) for r in resources))

        # Generate the graph
        g = self._create_graph(**graph_params)

        # For each unique resource, add them with their classes and their ancestors in the graph
        for uri, resource in resources_dict.items():
            self._complete_graph_with_resource(g, resource, resources_counter[uri], namespace_key)
        return g

    def _complete_graph_with_resource(self, graph, resource: DBpediaResource, resource_count: int,
                                      namespace_key: str = None):
        # Insert the resource as a node in the graph
        node_params = dict(count=resource_count, offset=resource.scores.offset)
        if self._concepts_information is not None and resource.uri in self._concepts_information:
            info = self._concepts_information[resource.uri]
            node_params['nbTypes'] = len(info.types)
            node_params['nbLinksIn'] = info.nb_links_in
            node_params['nbLinksOut'] = info.nb_links_out
        else:
            LOG.warning("uri info NOT found for %s" % resource.uri)
        rsc_node = self._write_resource_node(graph, resource, **node_params)

        # Extract the resource types
        types = self._get_resource_types(resource, namespace_key)
        LOG.debug("Type of %s: %s" % (str(resource.uri), str(types)))

        if types:
            # Process all types into the graph
            for rsc_type in types:
                # add the type as node with an edge from the resource
                rsc_type_node = self._write_type_node(graph, rsc_type)
                self._write_edge(graph, rsc_node, rsc_type_node)
                # iterate over the type's ancestors and write them in the graph
                for cl, parent in self._ontology_mgr.generate_ancestors(rsc_type, namespace_key):
                    cl_node = self._get_node_from_type(graph, cl)
                    parent_node = self._write_type_node(graph, parent)
                    self._write_edge(graph, cl_node, parent_node)
        else:
            # link the rsc to the abstract concept type of the manager
            self._write_edge(graph, rsc_node,
                             self._get_node_from_type(graph, self._ontology_mgr.get_abstract_concept_class()))

    def _get_resource_types(self, resource: DBpediaResource, namespace_key: str = None):
        # create a raw type generator
        types_gen = (t for t in resource.types)
        if self._concepts_information is not None and resource.uri in self._concepts_information:
            types_gen = chain(types_gen, (t for t in self._concepts_information[resource.uri].types))

        # surround the type generator with a mapping with URIRef transformation whose namespace is managed
        types_gen = map(lambda x: self._ontology_mgr.str_to_managed_uriref(x), types_gen)

        # filter to remove the None value
        types_gen = filter(lambda x: x is not None, types_gen)
        # if a namespace key is given, filter type with it
        if namespace_key is not None:
            types_gen = filter(lambda x: self._ontology_mgr.does_ref_belong_to_namespace(x, namespace_key), types_gen)
        # return a set of types
        return set(types_gen)

    @abstractmethod
    def _create_graph(self, **kwargs):
        pass

    @abstractmethod
    def _write_resource_node(self, graph, resource: DBpediaResource, **kwargs):
        """
        This method should create a node based on the resource, add all the nodes attributes given in kwargs,
        write the node within the grah and return the node.
        :param graph: the graph to write the node on.
        :param resource: the dbpedia resource.
        :param kwargs: the node's extra attributes
        :return: the created node.
        """
        pass

    @abstractmethod
    def _get_node_from_resource(self, graph, resource: DBpediaResource):
        pass

    @abstractmethod
    def _write_type_node(self, graph, cls: URIRef, **kwargs):
        """
        This method should create a node based on the class from an ontology, add all the nodes attributes given
        in kwargs, write the node within the grah and return the node.
        :param graph: the graph to write the node on.
        :param cls: the ontology class.
        :param kwargs: the node's extra attributes
        :return: the create node.
        """
        pass

    @abstractmethod
    def _get_node_from_type(self, graph, cls: URIRef):
        pass

    @abstractmethod
    def _write_edge(self, graph, node_1, node_2, **kwargs):
        """
        The method should create an edge between two nodes (node_1 -> node_2 if the graph is directed), and add the
        edge attributes given in kwargs.
        :param graph: the graph to write the edge on.
        :param node_1: the first node
        :param node_2: the second node
        :param kwargs: the edge's extra attributes
        :return: the create edge
        """
        pass

    @staticmethod
    @abstractmethod
    def to_json(filename: str, graph):
        pass


class NetworkXGraphBuilder(GraphBuilder):
    def __init__(self, ontology_manager: OntologyManager, concepts_information: Dict[str, ConceptInformation] = None):
        super().__init__(ontology_manager, concepts_information)

    @staticmethod
    def to_json(filename: str, graph: nx.Graph):
        json_graph = nx.node_link_data(graph)
        with open(filename, 'w') as f_out:
            json.dump(json_graph, f_out)

    @staticmethod
    def from_json(filename: str):
        with open(filename, 'r') as f_in:
            json_graph = json.load(f_in)
        return nx.node_link_graph(json_graph)

    def _create_graph(self, **kwargs) -> nx.Graph:
        if 'incoming_graph_data' in kwargs:  # just to secure the call
            del kwargs['incoming_graph_data']
        return nx.Graph(**kwargs)

    def _write_resource_node(self, graph, resource: DBpediaResource, **kwargs):
        """
        This method should create a node based on the resource, add all the nodes attributes given in kwargs,
        write the node within the grah and return the node.
        :param graph: the graph to write the node on.
        :param resource: the dbpedia resource.
        :param kwargs: the node's extra attributes
        :return: the created node.
        """
        graph.add_node(resource.uri, resource=True, **kwargs)
        return resource.uri

    def _get_node_from_resource(self, graph, resource: DBpediaResource):
        return resource.uri

    def _write_type_node(self, graph, cls: URIRef, **kwargs):
        """
        This method should create a node based on the class from an ontology, add all the nodes attributes
        given in kwargs, write the node within the grah and return the node.
        :param graph: the graph to write the node on.
        :param cls: the ontology class.
        :param kwargs: the node's extra attributes
        :return: the create node.
        """
        s_cls = str(cls)
        graph.add_node(s_cls, resource=False, **kwargs)
        return s_cls

    def _get_node_from_type(self, graph, cls: URIRef):
        return str(cls)

    def _write_edge(self, graph, node_1, node_2, **kwargs):
        """
        The method should create an edge between two nodes (node_1 -> node_2 if the graph is directed), and add the
        edge attributes given in kwargs.
        :param graph: the graph to write the edge on.
        :param node_1: the first node
        :param node_2: the second node
        :param kwargs: the edge's extra attributes
        :return: the create edge
        """
        return graph.add_edge(node_1, node_2)
