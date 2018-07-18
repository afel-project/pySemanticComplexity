# -*- coding: utf-8 -*-
"""Ontology for graph manipulation package. Provides classes to manage concepts RDF ontologies."""
import logging
from typing import Optional, List, Tuple, Union, Iterable

from rdflib import Graph as rdfGraph
from rdflib import Namespace, URIRef, RDFS

from utils.commons import ModuleShutUpWarning

__all__ = ['OntologyManager']

LOG = logging.getLogger(__name__)


class OntologyManager:
    _CONCEPT_URIREF = URIRef('#AbstractConcept#')

    def __init__(self):
        self._managed_namespaces = {}
        self._managed_ontologies_files = {}
        self._reference_graph = None

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
