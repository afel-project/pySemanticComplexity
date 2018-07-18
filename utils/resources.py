# -*- coding: utf-8 -*-
"""Resources constants."""
import logging.config
import os
from abc import ABCMeta
from collections import namedtuple
from typing import Dict, Set

from parsers.semantic.graphs.ontologies import OntologyManager

LOG = logging.getLogger(__name__)

__all__ = ['VENDOR_DIR_PATH', 'DefaultOntologies']

VENDOR_DIR_PATH = './vendor'

Ontology = namedtuple('Ontology', ['key', 'uri_base', 'filename', 'file_format'])


class DefaultOntologies(metaclass=ABCMeta):
    _AVAILABLE_ONTOLOGIES = {
        'DBPedia': Ontology(key='DBPedia', uri_base="http://dbpedia.org/ontology/",
                            filename=os.path.join(VENDOR_DIR_PATH, "dbpedia/dbpedia.nt"), file_format='nt'),
        'Schema': Ontology(key='Schema', uri_base="http://schema.org/",
                           filename=os.path.join(VENDOR_DIR_PATH, "dbpedia/schema.nt"), file_format='nt'),
        'yago': Ontology(key='yago', uri_base="http://dbpedia.org/class/yago/",
                         filename=os.path.join(VENDOR_DIR_PATH, "dbpedia/yago_taxonomy.ttl"), file_format='n3')
    }

    @classmethod
    def build_ontology_manager(cls, ontology_keys: Set[str] = None) -> OntologyManager:
        ontologies = cls._AVAILABLE_ONTOLOGIES.values() if not ontology_keys else \
            (cls._AVAILABLE_ONTOLOGIES[k] for k in ontology_keys)

        om = OntologyManager()
        for o in ontologies:
            om.add_ontology(key=o.key, namespace_uri=o.uri_base, filename=o.filename, file_format=o.file_format)
        return om.build_graph()

    @classmethod
    def available_ontologies(cls) -> Dict[str, str]:
        return dict((k, o.uri_base) for k, o in cls._AVAILABLE_ONTOLOGIES.items())
