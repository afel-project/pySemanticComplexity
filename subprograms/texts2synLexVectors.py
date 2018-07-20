# -*- coding: utf-8 -*-
import logging
from argparse import Namespace, ArgumentParser
from typing import Optional

import batchprocessing.synlex.stanfordSynLex as StanfordSynLex
from utils.commons import BatchProcess
from utils.stanfordResources import MemoryAllocationRule

LOG = logging.getLogger(__name__)

__all__ = ['Texts2synLexVectors']


class Texts2synLexVectors(BatchProcess):

    def _configure_args(self, parser: ArgumentParser) -> ArgumentParser:
        parser.description = "A batch processing for text to syntactical and lexical vectorization."
        parser.add_argument('data_in_dir', help='Texts input directory', metavar='<Texts Directory>',
                            type=str)
        parser.add_argument('out_file', help='output CSV file of vectors', metavar='<Output file>', type=str)
        parser.add_argument('-lex-type', help='lexical parsing type (default: bnc)', metavar='<Lexical Parser Type>',
                            choices=['anc', 'bnc'], type=str, default='bnc')
        parser.add_argument('-ei', '--ext-in', help='Extension of input files (default: ".txt")',
                            type=str, default='.txt')
        parser.add_argument('-nc', '--num-cores', help='Number of cores (default: 1)', type=int, default=1)
        parser.add_argument('--mem-lex', help='Memory allocated to the lex parser in Mo (default: 3000)',
                            type=int, default=3000)
        parser.add_argument('--mem-tregex', help='Memory allocated to the tregex parser in Mo (default: 100)',
                            type=int, default=100)
        parser.add_argument('--mem-postagger', help='Memory allocated to the pos tagger in Mo (default: 300)',
                            type=int, default=300)
        return parser

    def _run(self, args: Namespace) -> Optional[int]:
        # Setting memory allocation
        mar = MemoryAllocationRule()
        mar.postagger = args.mem_postagger
        mar.lexparser = args.mem_lex
        mar.tregex = args.mem_tregex

        self._logger.info("Start working...")
        dataset = StanfordSynLex.dir_to_vectors(args.data_in_dir, args.lex_type,
                                                args.num_cores, get_dataset=True, ext_in=args.ext_in)
        dataset.to_csv(args.out_file, index=False)
        self._logger.info("Work done.")
        return


if __name__ == '__main__':
    Texts2synLexVectors().start()
