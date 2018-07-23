# -*- coding: utf-8 -*-
import glob as glob
import logging
import os
from typing import Iterable, List, Union, Callable, Any

import pandas as pd
from sklearn.externals.joblib import Parallel, delayed

from parsers.semantic.graphs.tranformers import GraphTransformer
from utils.commons import safe_concurrency_backend, ModuleShutUpWarning

__all__ = ['compute_vector_from_graph', 'compute_vectors', 'compute_vectors_from_dir']

LOG = logging.getLogger(__name__)


def compute_vector_from_graph(graph, transformer: GraphTransformer):
    return transformer.vectorize_graph(graph)


def compute_vectors(graphs: Iterable, transformer: GraphTransformer, n_jobs: int,
                    backend: str = 'multiprocessing', to_dataset: bool = False) \
        -> Union[List[List[float]], pd.DataFrame]:
    backend = safe_concurrency_backend(backend)
    with ModuleShutUpWarning('rdflib'):
        vectors = Parallel(n_jobs=n_jobs, verbose=5, backend=backend)(
            delayed(compute_vector_from_graph)(g, transformer) for g in graphs
        )
    if to_dataset:
        return pd.DataFrame(vectors, columns=transformer.get_features_names())
    else:
        return vectors


def compute_vectors_from_dir(dir_in: str, graph_reader: Callable[[str], Any], transformer: GraphTransformer,
                             n_jobs: int, ext_in: str = ".json", backend: str = 'multiprocessing',
                             to_dataset: bool = False):
    backend = safe_concurrency_backend(backend)
    filepaths = list(glob.glob(os.path.join(dir_in, '*' + ext_in)))
    gen = (graph_reader(filepath) for filepath in filepaths)
    res = compute_vectors(gen, transformer, n_jobs, backend, to_dataset)
    if to_dataset:
        LOG.info("Adding filenames to dataset")
        filenames = [os.path.splitext(os.path.basename(filepath))[0] for filepath in filepaths]
        res.insert(0, 'filename', filenames)
    return res
