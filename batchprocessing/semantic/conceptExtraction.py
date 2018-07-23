# -*- coding: utf-8 -*-
import glob as glob
import logging
import os
import ujson as json
from typing import Iterable, List

from requests import RequestException
from sklearn.externals.joblib import Parallel, delayed

from parsers.preprocessing.text import TextPreprocessor
from parsers.semantic.dbpediaClients import DBpediaSpotlightClient
from parsers.semantic.model import TextConcepts
from utils.commons import safe_concurrency_backend

__all__ = ['generate_entities_from_paragraphs', 'text_to_entities', 'text_to_json_file', 'texts_to_entities',
           'dir_to_entities', 'dir_to_entities_json_files']

LOG = logging.getLogger(__name__)


def generate_entities_from_paragraphs(paragraphs: Iterable[str], client: DBpediaSpotlightClient):
    offset_span = 0
    for p in paragraphs:
        try:
            for entity in client.annotate(text=p):
                entity.scores.offset += offset_span
                yield entity
            offset_span += len(p)
        except RequestException as e:
            LOG.warning("Request Exception: %s" % str(e))


def text_to_entities(text: str, text_preprocessor: TextPreprocessor, client: DBpediaSpotlightClient,
                     is_filename: bool = False) -> TextConcepts:
    if is_filename:
        LOG.debug("Parsing concept of %s" % text)
        with open(text, 'r') as f_in:
            text = f_in.read()

    paragraphs = text_preprocessor.process_to_paragraphs(text)
    nb_words = sum(text_preprocessor.count_words(p) for p in paragraphs) if paragraphs else 0
    concepts = list(generate_entities_from_paragraphs(paragraphs, client))
    tc = TextConcepts(concepts, nb_words)

    return tc


def text_to_json_file(text: str, out_filename: str, text_preprocessor: TextPreprocessor,
                      client: DBpediaSpotlightClient, is_filename: bool = False):
    tc = text_to_entities(text, text_preprocessor, client, is_filename=is_filename)
    with open(out_filename, 'w') as f_out:
        json.dump(tc.to_dict(), f_out)


def texts_to_entities(texts: Iterable[str], text_preprocessor: TextPreprocessor,
                      client: DBpediaSpotlightClient, n_jobs: int, backend: str = 'multiprocessing') \
        -> List[TextConcepts]:
    backend = safe_concurrency_backend(backend, urllib_used=True)

    return Parallel(n_jobs=n_jobs, verbose=5, backend=backend)(
        delayed(text_to_entities)(text, text_preprocessor, client) for text in texts)


def dir_to_entities(dir_in: str, text_preprocessor: TextPreprocessor, client: DBpediaSpotlightClient,
                    n_jobs: int, backend: str = 'multiprocessing', in_ext: str = ".txt") -> List[TextConcepts]:
    backend = safe_concurrency_backend(backend, urllib_used=True)

    return Parallel(n_jobs=n_jobs, verbose=5, backend=backend)(
        delayed(text_to_entities)(filename, text_preprocessor, client, True)
        for filename in glob.glob(os.path.join(dir_in, '*' + in_ext)))


def dir_to_entities_json_files(dir_in: str, dir_out: str, text_preprocessor: TextPreprocessor,
                               client: DBpediaSpotlightClient, n_jobs: int, backend: str = 'multiprocessing',
                               force_rewrite: bool = False, in_ext: str = ".txt", out_ext: str = ".json"):
    backend = safe_concurrency_backend(backend, urllib_used=True)

    file_names = ((file_in, os.path.join(dir_out, os.path.splitext(os.path.basename(file_in))[0] + out_ext))
                  for file_in in glob.glob(os.path.join(dir_in, '*' + in_ext)))
    if not force_rewrite:
        file_names = filter(lambda x: not os.path.exists(x[1]), file_names)

    Parallel(n_jobs=n_jobs, verbose=5, backend=backend)(
        delayed(text_to_json_file)(f[0], f[1], text_preprocessor, client, True) for f in file_names)
