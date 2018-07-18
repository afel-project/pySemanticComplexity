# -*- coding: utf-8 -*-
"""Batch utility to convert text files directly into concepts vectors of semantic complexity."""
import glob
import logging
import os
import ujson as json
from abc import ABCMeta
from argparse import Namespace, ArgumentParser
from typing import Set, Optional

import batchprocessing.semantic.conceptExtraction as ConceptsExtraction
import batchprocessing.semantic.conceptsEnrichment as ConceptsEnrichment
import batchprocessing.semantic.graphCreation as GraphCreation
import batchprocessing.semantic.graphVectorization as GraphVectorization
from parsers.preprocessing.text import TextPreprocessor
from parsers.semantic.dbpediaClients import EntitiesTypesRetriever, DBpediaSpotlightClient, \
    LinksCountEntitiesRetriever
from parsers.semantic.graphs.builders import NetworkXGraphBuilder
from parsers.semantic.graphs.tranformers import NamespaceNetworkxGraphTransformer
from utils.commons import BatchProcess, file_can_be_write
from utils.resources import DefaultOntologies

LOG = logging.getLogger(__name__)

__all__ = ['Texts2Vectors']


class Texts2VectorsRunner(metaclass=ABCMeta):

    @classmethod
    def process_files_to_vectors(cls, dir_in: str, out_file: str, spotlight_ep: str, num_cores: int, ext_in: str,
                                 confidence: float, sparql_ep: str, max_concepts: int, nice_server: bool,
                                 concept_info_filename: str, ontology_keys: Set[str], graphs_dir: str,
                                 backend: str = "multiprocessing"):
        # Before going further, test the existence of dirs, and the possibility to write documents
        LOG.info("1/7: Checking directories and files rights...")
        cls._check_rights(dir_in, out_file, concept_info_filename, graphs_dir)

        # Create the different composants required to the transformation
        LOG.info("2/7: Create transformer component")
        text_processor = TextPreprocessor()  # Preprocessing of texts
        client = DBpediaSpotlightClient(spotlight_ep, confidence=confidence)  # Spotlight client
        types_retriever = EntitiesTypesRetriever(sparql_endpoint=sparql_ep, max_entities_per_query=max_concepts,
                                                 nice_to_server=nice_server)  # types retriever
        links_retriever = LinksCountEntitiesRetriever(sparql_endpoint=sparql_ep, max_entities_per_query=50,
                                                      nice_to_server=nice_server)  # types retriever
        LOG.info("Creating ontology manager (Can take time...)")
        ontology_manager = DefaultOntologies.build_ontology_manager(ontology_keys)
        LOG.info("Ontology manager will used these ontologies: %s" % str(ontology_keys))
        graph_transformer = NamespaceNetworkxGraphTransformer(ontology_manager.managed_namespaces)  # Graph transformer

        # Create a list of text files and a list of simple text file name that will be used in different functions
        input_files_names = [os.path.splitext(os.path.split(file)[1])[0] for file in
                             glob.glob(os.path.join(dir_in, '*' + ext_in))]

        # Compute entities of each files
        LOG.info("3/7: Clean texts and detect DBpedia entities from them...")
        # Treat the files in parallel to get a list of list of DBpediaResource
        texts_concepts = ConceptsExtraction.dir_to_entities(dir_in, text_processor, client, num_cores, backend,
                                                            in_ext=ext_in)

        # Compute a entity - type mapping set
        LOG.info("4/7: Retrieve types for all entities...")
        concepts_info = ConceptsEnrichment.get_concepts_information(texts_concepts, types_retriever, links_retriever)

        # Save the mapping if requiring
        if concept_info_filename is not None:
            LOG.info("Save the concept - info mapping...")
            with open(concept_info_filename, 'w') as f:
                json.dump(concepts_info, f)

        # Create graphs for each list of entities (and save them if required)
        LOG.info("5/7: Create graphs of concept for each texts, and vectorize them...")
        # If saving graph is required, create a list of out files
        if graphs_dir is not None:
            LOG.info("Will save the graph files...")
            out_graph_files = [os.path.join(graphs_dir, filename + '.json') for filename in input_files_names]
        else:
            out_graph_files = []
        # Create the graphs based
        graph_builder = NetworkXGraphBuilder(ontology_manager, concepts_info)
        graphs = GraphCreation.compute_graphs(texts_concepts, graph_builder, num_cores, backend, out_graph_files)

        LOG.info("6/7: Vectorize graphs")
        dataset = GraphVectorization.compute_vectors(graphs, graph_transformer, num_cores, backend, to_dataset=True)
        # Insert the filename column into the dataset
        dataset.insert(0, 'filename', input_files_names)

        # Save the dataframe
        LOG.info("7/7: Save the vectors")
        dataset.to_csv(out_file, index=False)

    @classmethod
    def _check_rights(cls, dir_texts_in: str, out_file: str, concept_info_filename: str, graphs_dir: str):
        # Test if dir_texts_in exists and is readble
        if not os.access(dir_texts_in, os.R_OK):
            raise ValueError("Text directory does not exist or is not readable.")
        # Test if out_file is writable or does not exists but its directory is writable
        if not file_can_be_write(out_file):
            raise ValueError("CSV output file is not writable or cannot be created.")
        # If concept_infofilename is given, test if it is writable or does not exists but its directory is writable
        if concept_info_filename is not None and not file_can_be_write(concept_info_filename):
            raise ValueError("Concept-info mapping file is not writable or cannot be created.")
        # If graphs_dir is given, test if the directory is writable
        if graphs_dir is not None and not os.access(graphs_dir, os.W_OK):
            raise ValueError("Graphs directory does not exist or is not writable.")


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
        parser.add_argument('--save-info',
                            help='Save the entity-fino mapping dictionnary to a json file (default: No)',
                            metavar='<Json File>', type=str, default=None)
        parser.add_argument('-on', '--ontology', help='Ontologies to used (Default: all)', type=str, action='append',
                            choices=DefaultOntologies.available_ontologies().keys(), default=[])
        parser.add_argument('--save-graphs', help='Save the several graphs as json files into a directory',
                            metavar='<Graphs Directory>', type=str, default=None)
        return parser

    def _run(self, args: Namespace) -> Optional[int]:
        Texts2VectorsRunner.process_files_to_vectors(args.data_in_dir, args.out_file, args.spotlight_ep,
                                                     args.num_cores, args.ext_in, args.confidence, args.sparql_ep,
                                                     args.max_concepts, args.nice_server, args.save_info,
                                                     args.ontology, args.save_graphs)
        return
