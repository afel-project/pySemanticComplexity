# -*- coding: utf-8 -*-
import glob
import os
import logging
from typing import Union

import numpy as np
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed

from parsers.lexical.stanford import StanfordLexicalTransformer, ANCStanfordLexicalTransformer, \
    BNCStanfordLexicalTransformer
from parsers.preprocessing.text import TextPreprocessor
from parsers.syntactic.stanford import StanfordSyntacticTransformer
from utils.commons import safe_concurrency_backend

LOG = logging.getLogger(__name__)

__all__ = ['file_to_vector', 'get_full_features_names', 'dir_to_vectors']


def file_to_vector(filename: str, text_preprocessor: TextPreprocessor,
                   syntactic_transformer: StanfordSyntacticTransformer,
                   lexical_transformer: StanfordLexicalTransformer) -> np.ndarray:
    LOG.info("Cleaning text")
    with open(filename, 'r') as f:
        text = "\n".join(text_preprocessor.process_to_paragraphs(f.read()))
    LOG.info("Computing syntaxtic features...")
    syntactic_features = syntactic_transformer.compute_features(text)
    LOG.info("Computing lexical features")
    lexical_features = lexical_transformer.compute_features(text)
    return np.hstack([syntactic_features, lexical_features])


def get_full_features_names(syntactic_transformer: StanfordSyntacticTransformer,
                            lexical_transformer: StanfordLexicalTransformer) -> np.ndarray:
    return np.hstack([syntactic_transformer.get_features(), lexical_transformer.get_features()])


def dir_to_vectors(dir_in: str, lexical_parser: str, nb_jobs: int, get_dataset: bool = True,
                   ext_in: str = ".txt", backend: str = "multiprocessing") -> Union[np.ndarray, pd.DataFrame]:
    backend = safe_concurrency_backend(backend)
    file_names = glob.glob(os.path.join(dir_in, '*' + ext_in))

    syntactic_transformer = StanfordSyntacticTransformer(file_names=False)
    if lexical_parser == 'anc':
        lexical_transformer = ANCStanfordLexicalTransformer(file_names=False)
    elif lexical_parser == 'bnc':
        lexical_transformer = BNCStanfordLexicalTransformer(file_names=False)
    else:
        raise TypeError("Unknown lexical parser type '%s'" % lexical_parser)
    text_preprocessor = TextPreprocessor()

    LOG.info("Creating vectors")
    vectors = Parallel(n_jobs=nb_jobs, verbose=5, backend=backend)(
        delayed(file_to_vector)(f, text_preprocessor, syntactic_transformer, lexical_transformer) for f in file_names
    )
    if get_dataset:
        LOG.debug("Creating dataset")
        dataset = pd.DataFrame(vectors, columns=get_full_features_names(syntactic_transformer, lexical_transformer))
        LOG.debug("Adding filenames to dataset")
        dataset.insert(0, 'filename', [os.path.splitext(os.path.basename(f))[0] for f in file_names])
        return dataset
    else:
        return np.array(vectors)
