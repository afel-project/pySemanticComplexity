# -*- coding: utf-8 -*-
"""Batch utility to convert text files directly into concepts vectors of semantic complexity."""
import gc
import glob
import logging
import os
import ujson as json
from abc import ABCMeta
from argparse import Namespace, ArgumentParser
from itertools import zip_longest
from typing import Set, Tuple, Dict, List, Optional

import pandas as pd
from requests.exceptions import RequestException
from sklearn.externals.joblib import Parallel, delayed

from dbpedia.concept import ConceptTypeRetriever
from dbpedia.entities import DBpediaResource
from dbpedia.graphs import GraphBuilderFactory, NetworkXGraphBuilder, GraphTransformer
from dbpedia.spotlight import DBpediaSpotlightClient
from utils.commons import BatchProcess, file_can_be_write
from utils.filePreprocessor import TextPreprocessor

LOG = logging.getLogger(__name__)

__all__ = ['Texts2VectorsRunner']


class Texts2VectorsRunner(metaclass=ABCMeta):

    @classmethod
    def process_files_to_vectors(cls, dir_in: str, out_file: str, spotlight_ep: str, num_cores: int, ext_in: str,
                                 confidence: float, sparql_ep: str, max_concepts: int, nice_server: bool,
                                 concept_type_filename: str, graphs_dir: str, backend: str = "threading"):
        # Before going further, test the existence of dirs, and the possibility to write documents
        LOG.info("1/5: Checking directories and files rights...")
        cls._check_rights(dir_in, out_file, concept_type_filename, graphs_dir)

        # Create a list of text files and a list of simple text file name that will be used in different functions
        input_files = list(glob.glob(os.path.join(dir_in, '*' + ext_in)))
        input_files_names = [os.path.splitext(os.path.split(file)[1])[0] for file in input_files]

        # Compute entities of each files
        LOG.info("2/5: Clean texts and detect DBpedia entities from them...")
        entities_list, entity_set = cls._process_files_to_entities(input_files, spotlight_ep, confidence, num_cores,
                                                                   backend)

        # Compute a entity - type mapping set
        LOG.info("3/5: Retrieve types for all entities...")
        concepts_types = cls._retrieve_concepts_types(entity_set, sparql_ep, max_concepts, nice_server)

        # Save some memory: delete the entity_set and the input_files list
        LOG.info("Memory cleaning...")
        del entity_set, input_files
        gc.collect()

        # Save the mapping if requiring
        if concept_type_filename is not None:
            LOG.info("Save the entity - types mapping...")
            with open(concept_type_filename, 'w') as f:
                json.dump(concepts_types, f)

        # Create graphs for each list of entities (and save them if required)
        LOG.info("4/5: Create graphs of concept for each texts, and vectorize them...")
        # If saving graph is required, create a list of out files
        if graphs_dir is not None:
            LOG.info("Will save the graph files...")
            out_graph_files = [os.path.join(graphs_dir, filename + '.json') for filename in input_files_names]
        else:
            out_graph_files = []
        dataset = cls._compute_graphs_and_vectors(entities_list, out_graph_files, concepts_types, num_cores, backend)

        # Insert the filename column into the dataset
        dataset.insert(0, 'filename', input_files_names)

        # Save some memory: delete entities_list, input_files_names
        LOG.info("Memory cleaning...")
        del entities_list, input_files_names
        gc.collect()

        # Save the dataframe
        LOG.info("5/5: Save the vectors")
        dataset.to_csv(out_file, index=False)

    @classmethod
    def _check_rights(cls, dir_texts_in: str, out_file: str, concept_type_filename: str, graphs_dir: str):
        # Test if dir_texts_in exists and is readble
        if not os.access(dir_texts_in, os.R_OK):
            raise ValueError("Text directory does not exist or is not readable.")
        # Test if out_file is writable or does not exists but its directory is writable
        if not file_can_be_write(out_file):
            raise ValueError("CSV output file is not writable or cannot be created.")
        # If concept_type_filename is given, test if it is writable or does not exists but its directory is writable
        if concept_type_filename is not None and not file_can_be_write(concept_type_filename):
            raise ValueError("Concept-types mapping file is not writable or cannot be created.")
        # If graphs_dir is given, test if the directory is writable
        if graphs_dir is not None and not os.access(graphs_dir, os.W_OK):
            raise ValueError("Graphs directory does not exist or is not writable.")

    @classmethod
    def _process_files_to_entities(cls, files_in: List[str], spotlight_ep: str, confidence: float,
                                   num_cores: int, backend: str) \
            -> Tuple[List[List[DBpediaResource]], Set[DBpediaResource]]:
        """Process in parrell each texts to obtain a list of list of DBPediaResource (one list for each text)
        and a set of DBPediaResource (common for all texts"""
        # Create the text preprocessor and the Spotligh REST Client
        text_processor = TextPreprocessor()
        client = DBpediaSpotlightClient(spotlight_ep)

        # Treat the files in parallel to get a list of list of DBpediaResource
        entities_list = Parallel(n_jobs=num_cores, verbose=5, backend=backend)(
            delayed(cls._process_file_to_entities)(f, text_processor, client, confidence) for f in files_in)

        # Create a set of DBPedia resources
        entities_set = set(e for l in entities_list for e in l)

        return entities_list, entities_set

    @classmethod
    def _process_file_to_entities(cls, filename, text_processor, client: DBpediaSpotlightClient, confidence: float) \
            -> List[DBpediaResource]:
        """Process a file to clean it, split it in paragraphs, then retrieve all DBPedia resources for all paragraphs
        And merge them into a unique list"""
        with open(filename, 'r') as f_in:
            paragraphs = text_processor.process_to_paragraphs(f_in.read())
        LOG.debug("Retrieve concept of %s" % filename)
        concepts = cls._paragraphs_to_entities(paragraphs, client, confidence)
        return concepts

    @classmethod
    def _paragraphs_to_entities(cls, paragraphs: List[str], client: DBpediaSpotlightClient, confidence: float) \
            -> List[DBpediaResource]:
        try:
            return [concept[0] for p in paragraphs for concept in client.annotate(text=p, confidence=confidence)]
        except RequestException as e:
            LOG.warning("Request Exception: %s" % str(e))
            return []

    @classmethod
    def _retrieve_concepts_types(cls, entity_set: Set[DBpediaResource], sparql_ep: str,
                                 max_concepts: int, nice_server: bool) -> Dict[str, List[str]]:
        """Retrieve all types of all resources through a sequence of SPARQL requests."""
        # Create the types retriever
        retriever = ConceptTypeRetriever(sparql_ep, max_concepts, nice_server)
        concepts_types = retriever.retrieve_resource_types(entity_set)
        return concepts_types

    @classmethod
    def _compute_graphs_and_vectors(cls, entities_lists: List[List[DBpediaResource]], out_files: List[str],
                                    concept_types: Dict[str, List[str]],
                                    num_cores: int, backend: str) -> pd.DataFrame:
        """Create graph for each entities lists corresponding to each text files.
        Save them in json files if required."""
        # Create the graph builder based on default ontologies and the concepts-types dictionnary
        # and the graph transformer
        factory = GraphBuilderFactory()
        graph_builder = factory.build_networkx_graph_builer(concepts_types=concept_types)
        graph_transformer = GraphTransformer()

        # Create graph and vectors in parrallel of each couple entities - out_file (out_file might be null)
        vectors = Parallel(n_jobs=num_cores, verbose=5, backend=backend)(
            delayed(cls._compute_graph_and_vectors)(entities, out_file, graph_builder, graph_transformer)
            for entities, out_file, in zip_longest(entities_lists, out_files)
        )

        # Create the dataset based on graph
        LOG.debug("Create the dataset...")
        dataset = pd.DataFrame(vectors, columns=graph_transformer.get_features_names())

        return dataset

    @classmethod
    def _compute_graph_and_vectors(cls, entities: List[DBpediaResource], out_file: str,
                                   graph_builder: NetworkXGraphBuilder, transformer: GraphTransformer) -> List[float]:
        """Create a graph for an entities list and save it in out_file it required, then vectorize if"""
        # Create the graph of concepts
        graph = graph_builder.build_graph_from_entities(entities)
        # Save if if required
        if out_file is not None:
            LOG.debug("Save the graph %s" % out_file)
            graph_builder.to_json(out_file, graph)
        # Vectorize the graph
        return transformer.vectorize_graph(graph)


class Texts2Vectors(BatchProcess):
    def _configure_args(self, parser: ArgumentParser) -> ArgumentParser:
        # Description of the sub program
        parser.description = "A parallel implementation of the full pipeline of translation from texts to vectors " \
                             "of semantic complexity."
        parser.epilog = """For each text file processed in parallel, the pipeline first clean and split text in paragraphs, 
and identify the DBpedia entity with the use of a Spotlight REST Api. Each entity is then enrich with its types by 
queriyng a DBpedia SPARQL endpoint. Each list of entities (for each document) is then processed in parallel to compute
a graph of concept, conposed of the entities, their types and the hierachy of ontology classes that define these types.
Finally each graph is vectorized in parallel. Note that three ontologies are used so far: Schema, DBpedia ontology and
Yago. The result csv file is composed one line per fil, each of them being composed of the file name (without its
extension) and the several complexity features."""

        # Positionnal arguments
        parser.add_argument('data_in_dir', help='Texts input directory', metavar='<Texts Directory>',
                            type=str)
        parser.add_argument('out_file', help='output CSV file of vectors', metavar='<Output file>', type=str)
        parser.add_argument('spotlight_ep', help='DBpedia spotlight endpoint '
                                                 '(ex.: "https://api.dbpedia-spotlight.org/en/annotate")',
                            metavar='<Spotlight Endpoint>', type=str)
        # Optional arguments
        parser.add_argument('-nc', '--num-cores', help='Number of cores to use for parallel processing (default: 1)',
                            type=int, default=1)
        parser.add_argument('-ei', '--ext-in', help='Extension of input files (default: ".txt")',
                            type=str, default='.txt')
        parser.add_argument('-co', '--confidence', help='Confidence of the DBpedia spotlight service (default: 0.5).',
                            type=float, default=0.5)
        parser.add_argument('--sparql-ep', help='SPARQL endpoint (default: "http://dbpedia.org/sparql")',
                            type=str, default='http://dbpedia.org/sparql')
        parser.add_argument('--max-concepts', help='Max concepts per SPARQL query (default: 100)',
                            type=int, default=100)
        parser.add_argument('--nice-server', help='Be nice to the SPARQL server by waiting randomly from 0 to 1 '
                                                  'second between each request (disabled by default, but recommended)',
                            action='store_false')
        parser.add_argument('--save-types',
                            help='Save the entity-types mapping dictionnary to a json file (default: No)',
                            metavar='<Json File>', type=str, default=None)
        parser.add_argument('--save-graphs', help='Save the several graphs as json files into a directory',
                            metavar='<Graphs Directory>', type=str, default=None)
        return parser

    def _run(self, args: Namespace) -> Optional[int]:
        Texts2VectorsRunner.process_files_to_vectors(args.data_in_dir, args.out_file, args.spotlight_ep,
                                                     args.num_cores, args.ext_in, args.confidence, args.sparql_ep,
                                                     args.max_concepts, args.nice_server, args.save_types,
                                                     args.save_graphs)
        return
