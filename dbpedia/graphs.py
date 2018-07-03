# -*- coding: utf-8 -*-
"""Concept graph manipulation package. Provides classes to manage concepts RDF ontologies, concepts graphs and
to operate on graphs."""
import logging
import ujson as json
from abc import abstractmethod, ABCMeta
from collections import Counter
from itertools import chain
from typing import Optional, List, Tuple, Union, Set, Iterable, Dict

import networkx as nx
from rdflib import Graph as rdfGraph
from rdflib import Namespace, URIRef, RDFS

from utils.commons import AVAILABLE_ONTOLOGIES, Ontology, ModuleShutUpWarning
from .entities import DBpediaResource

__all__ = ['OntologyManager', 'GraphBuilder', 'NetworkXGraphBuilder', 'GraphBuilderFactory']

LOG = logging.getLogger(__name__)


class OntologyManager:
    _CONCEPT_URIREF = URIRef('#AbstractConcept#')

    def __init__(self):
        self._managed_namespaces: Dict[str, Namespace] = {}
        self._managed_ontologies_files: Dict[str, Tuple[str, str]] = {}
        self._reference_graph: rdfGraph = None

    def get_ontology_keys(self) -> Iterable[str]:
        return list(self._managed_namespaces.keys())

    def add_ontology(self, key: str, namespace_uri: str, filename: str, file_format: str = 'nt') -> 'OntologyManager':
        self._managed_namespaces[key] = Namespace(namespace_uri)
        self._managed_ontologies_files[key] = (filename, file_format)
        return self

    def remove_ontology(self, key: str) -> 'OntologyManager':
        if key in self._managed_namespaces:
            del self._managed_namespaces[key]
            del self._managed_ontologies_files[key]
        return self

    def build_graph(self) -> 'OntologyManager':
        g = rdfGraph()
        for key, namespace in self._managed_namespaces.items():
            g.bind(key, namespace)
        with ModuleShutUpWarning('rdflib'):
            for filename, file_format in self._managed_ontologies_files.values():
                try:
                    g.parse(filename, format=file_format)
                except FileNotFoundError as e:
                    LOG.fatal("Missing onotlogy file '%s'" % filename)
                    raise e
        self._reference_graph = g
        return self

    def str_to_managed_uriref(self, ref: str, namespace_key: str = None) -> Optional[Union[str, URIRef]]:
        """
        If namespace_key is given, attemps to provide the suffix of the ref is it belongs to the namespace,
        None otherwise. If namespace_key is None, return the best candidate as a URIRef: an UriRef whose the namespace
        prefix fits the ref and the suffix is the smallest of all candidates. Return None if no candidates have been
        found.
        :param ref: the uri or the qname
        :param namespace_key: the key of the namespace
        :return: None or the URIRef if namespace_key is None, None or ref suffix otherwise
        """
        if namespace_key is not None:
            namespace = self._managed_namespaces[namespace_key]
            # Test if ref starts with the namespace key and :
            if ref.startswith(namespace_key + ':'):
                return ref[len(namespace_key) + 1:]
            elif ref.startswith(namespace):
                return ref[len(namespace):]
            else:
                return None
        else:
            # test ref with all namespaces
            candidates = ((namespace, self.str_to_managed_uriref(ref, ns_key))
                          for ns_key, namespace in self._managed_namespaces.items())
            # Keep candidates only
            candidates = filter(lambda x: x[1] is not None, candidates)
            # Sort candidates by the length and take the first one
            candidates = sorted(candidates, key=lambda x: len(x[1]))
            if candidates:
                namespace, suffix = candidates[0]
                return namespace.term(suffix)
            else:
                return None

    def does_ref_belong_to_namespaces(self, ref: str) -> bool:
        return self.str_to_managed_uriref(ref) is not None

    def does_ref_belong_to_namespace(self, ref: str, namespace_key: str) -> bool:
        return self.str_to_managed_uriref(ref, namespace_key) is not None

    def generate_parents(self, cl: URIRef, namespace_key: str = None) -> [URIRef]:
        parents = self._reference_graph.objects(cl, RDFS.subClassOf)
        if namespace_key is not None:
            parents = filter(lambda o: isinstance(o, URIRef) and o.startswith(self._managed_namespaces[namespace_key]),
                             parents)
        has_parents = False
        for parent in parents:
            has_parents = True
            yield parent
        if not has_parents:
            yield self._CONCEPT_URIREF

    def generate_ancestors(self, cl: URIRef, namespace_key: str = None) -> List[Tuple[URIRef, URIRef]]:
        for parent in self.generate_parents(cl, namespace_key):
            # LOG.debug("%s -> %s" % (cl, parent))
            yield (cl, parent)
            if parent != self._CONCEPT_URIREF:
                for ancestor_tuple in self.generate_ancestors(parent, namespace_key):
                    yield ancestor_tuple

    def get_namespace(self, key: str) -> Namespace:
        return self._managed_namespaces.get(key, None)

    @property
    def reference_graph(self) -> rdfGraph:
        return self._reference_graph

    @classmethod
    def get_abstract_concept_class(cls) -> URIRef:
        return cls._CONCEPT_URIREF

    @property
    def managed_namespaces(self):
        return self._managed_namespaces

    @classmethod
    def get_root(cls):
        return cls._CONCEPT_URIREF


class GraphBuilder(metaclass=ABCMeta):
    def __init__(self, ontology_manager: OntologyManager, concepts_types: dict = None):
        self._ontology_mgr = ontology_manager
        self.concepts_types = concepts_types

    @property
    def ontology_manager(self) -> OntologyManager:
        return self._ontology_mgr

    @ontology_manager.setter
    def ontology_manager(self, value: OntologyManager):
        self._ontology_mgr = value

    @ontology_manager.deleter
    def ontology_manager(self):
        raise AttributeError("Cannot delete ontology manager.")

    def build_graph_from_entities(self, resources: [DBpediaResource]):
        return self.build_sub_graph_from_entities(resources, namespace_key=None)

    def build_sub_graph_from_entities(self, resources: [DBpediaResource], namespace_key: str = None):
        LOG.debug("Build a graph for %d resources" % len(resources))
        if namespace_key is not None:
            # Check that the namespace key exists in the ontology manager
            assert self._ontology_mgr.get_namespace(namespace_key) is not None

        # Create a counter of resources, and a dict of uri-resource to have a unique liste of resources
        resources_counter = Counter((r.uri for r in resources))
        resources_dict = dict(((r.uri, r) for r in resources))

        # Generate the graph
        g = self._create_graph()

        # For each unique resource, add them with their classes and their ancestors in the graph
        for uri, resource in resources_dict.items():
            self._complete_graph_with_resource(g, resource, resources_counter[uri], namespace_key)
        return g

    def _complete_graph_with_resource(self, graph, resource: DBpediaResource, resource_count: int,
                                      namespace_key: str = None):
        # Insert the resource as a node in the graph
        rsc_node = self._write_resource_node(graph, resource, count=resource_count, offset=resource.scores.offset)

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
        if self.concepts_types is not None and resource.uri in self.concepts_types:
            types_gen = chain(types_gen, (t for t in self.concepts_types[resource.uri]))

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
    def _create_graph(self):
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


class NetworkXGraphBuilder(GraphBuilder):
    def __init__(self, ontology_manager: OntologyManager, concepts_types: dict = None):
        super().__init__(ontology_manager, concepts_types)

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

    def _create_graph(self, **kwargs):
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


class Singleton(type):
    def __init__(cls, *args, **kwargs):
        cls.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__call__(*args, **kwargs)
        return cls.__instance


class GraphBuilderFactory(metaclass=Singleton):
    def __init__(self):
        self._default_om = None

    @property
    def default_ontology_manager(self) -> OntologyManager:
        return self._default_om

    @default_ontology_manager.setter
    def default_ontology_manager(self, ontology_manager: OntologyManager):
        self._default_om = ontology_manager

    @default_ontology_manager.deleter
    def default_ontology_manager(self):
        raise ValueError("Cannot delete ontology manager from the graph builder factory.")

    def build_networkx_graph_builer(self, ontology_manager: OntologyManager = None, concepts_types: dict = None) \
            -> NetworkXGraphBuilder:
        if ontology_manager is None:
            if self._default_om is None:
                raise ValueError("Cannot build graph builder: ontology manager missing")
            else:
                ontology_manager = self._default_om
        return NetworkXGraphBuilder(ontology_manager, concepts_types)

    @staticmethod
    def get_available_default_ontology_keys():
        return [o.key for o in AVAILABLE_ONTOLOGIES]

    @staticmethod
    def build_default_ontology_manager(ontology_keys: Set[str] = None) -> OntologyManager:
        ontologies = AVAILABLE_ONTOLOGIES if not ontology_keys \
            else [o for o in AVAILABLE_ONTOLOGIES if o.key in ontology_keys]
        om = OntologyManager()
        for o in ontologies:
            om.add_ontology(key=o.key, namespace_uri=o.uri_base, filename=o.filename, file_format=o.file_format)
        return om.build_graph()

    @staticmethod
    def build_ontology_manager(ontologies: Set[Ontology]) -> OntologyManager:
        om = OntologyManager()
        for o in ontologies:
            om.add_ontology(key=o.key, namespace_uri=o.uri_base, filename=o.filename, file_format=o.file_format)
        return om.build_graph()
