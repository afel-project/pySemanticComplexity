# -*- coding: utf-8 -*-
"""Batch utility to convert text files into concepts json files."""
import logging
from argparse import Namespace, ArgumentParser
from typing import Optional

import batchprocessing.semantic.conceptExtraction as ConceptsExtraction
from parsers.preprocessing.text import TextPreprocessor
from parsers.semantic.dbpediaClients import DBpediaSpotlightClient
from utils.commons import BatchProcess

LOG = logging.getLogger(__name__)

__all__ = ['Texts2Concepts']


class Texts2Concepts(BatchProcess):

    def _configure_args(self, parser: ArgumentParser) -> ArgumentParser:
        parser.description = "A parallel implementation of batch processing for text to DBpedia concepts translation."
        parser.add_argument('data_in_dir', help='Texts input directory', metavar='<Texts Directory>',
                            type=str)
        parser.add_argument('data_out_dir', help='Data output directory', metavar='<Concepts Directory>',
                            type=str)
        parser.add_argument('endpoint', help='DBpedia spotlight endpoint '
                                             '(ex.: "https://api.dbpedia-spotlight.org/en/annotate")',
                            metavar='<Spotlight Endpoint>', type=str)
        parser.add_argument('-co', '--confidence', help='Confidence of the DBpedia spotlight service (default: 0.5).',
                            type=float, default=0.5)
        parser.add_argument('-ei', '--ext-in', help='Extension of input files (default: ".txt")',
                            type=str, default='.txt')
        parser.add_argument('-f', '--force', help='Do not take care of already existing output files '
                                                  '(default: disabled)', action='store_true')
        parser.add_argument('-nc', '--num-cores', help='Number of cores (default: 1)', type=int, default=1)
        return parser

    def _run(self, args: Namespace) -> Optional[int]:
        client = DBpediaSpotlightClient(args.endpoint, confidence=args.confidence)
        text_processor = TextPreprocessor()
        self._logger.info("Start working...")
        ConceptsExtraction.dir_to_entities_json_files(args.data_in_dir, args.data_out_dir, text_processor, client,
                                                      args.num_cores, force_rewrite=args.force, in_ext=args.ext_in)
        self._logger.info("Work done.")
        return


if __name__ == '__main__':
    Texts2Concepts().start()
