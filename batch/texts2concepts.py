# -*- coding: utf-8 -*-
"""Batch utility to convert text files into concepts json files."""
import glob
import logging
import os
import ujson as json
from abc import ABCMeta
from argparse import Namespace, ArgumentParser
from typing import Optional, List

from requests.exceptions import RequestException
from sklearn.externals.joblib import Parallel, delayed

from dbpedia.entities import DBpediaResource
from dbpedia.spotlight import DBpediaSpotlightClient
from utils.commons import BatchProcess
from utils.filePreprocessor import TextPreprocessor

LOG = logging.getLogger(__name__)

__all__ = ['Texts2Concepts', 'Texts2ConceptsRunner']


class Texts2ConceptsRunner(metaclass=ABCMeta):

    @classmethod
    def process_file_to_entities(cls, filename, text_processor: TextPreprocessor, client: DBpediaSpotlightClient,
                                 confidence: float) -> List[DBpediaResource]:
        """Process a file to clean it, split it in paragraphs, then retrieve all DBPedia resources for all paragraphs
        And merge them into a unique list"""
        with open(filename, 'r') as f_in:
            paragraphs = text_processor.process_to_paragraphs(f_in.read())
        LOG.debug("Retrieve concept of %s" % filename)
        concepts = list(cls.paragraphs_to_entities(paragraphs, client, confidence))
        return concepts

    @classmethod
    def paragraphs_to_entities(cls, paragraphs: List[str], client: DBpediaSpotlightClient, confidence: float) \
            -> List[DBpediaResource]:

        offset_span = 0
        for p in paragraphs:
            try:
                for entity in client.annotate(text=p, confidence=confidence):
                    entity.scores.offset += offset_span
                    yield entity
                offset_span += len(p)
            except RequestException as e:
                LOG.warning("Request Exception: %s" % str(e))

    @classmethod
    def dir_to_concept_json_files(cls, dir_in, dir_out, client: DBpediaSpotlightClient, text_processor,
                                  confidence, in_ext, n_jobs, force, backend: str = "threading"):
        file_names = ((file_in, os.path.join(dir_out, os.path.splitext(os.path.basename(file_in))[0] + '.json'))
                      for file_in in glob.glob(os.path.join(dir_in, '*' + in_ext)))

        if not force:
            file_names = filter(lambda x: not os.path.exists(x[1]), file_names)

        Parallel(n_jobs=n_jobs, verbose=5, backend=backend)(
            delayed(cls._file_to_concept_json_file)(f[0], f[1], client, text_processor, confidence) for f in file_names)

    @classmethod
    def _file_to_concept_json_file(cls, file_in, file_out, client: DBpediaSpotlightClient, text_processor, confidence):
        LOG.info("Parsing concept of %s" % file_in)
        concepts = cls.process_file_to_entities(file_in, text_processor, client, confidence=confidence)
        with open(file_out, 'w') as f_out:
            json.dump([c.to_dict() for c in concepts], f_out)


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
        client = DBpediaSpotlightClient(args.endpoint)
        text_processor = TextPreprocessor()
        self._logger.info("Start working...")
        Texts2ConceptsRunner \
            .dir_to_concept_json_files(args.data_in_dir, args.data_out_dir, client, text_processor, args.confidence,
                                       args.ext_in, args.num_cores, args.force)
        self._logger.info("Work done.")
        return


if __name__ == '__main__':
    Texts2Concepts().start()
