# -*- coding: utf-8 -*-
import glob
import logging
import os
import ujson as json
from itertools import zip_longest
from typing import Iterable, Any

from sklearn.externals.joblib import Parallel, delayed

from parsers.semantic.graphs.builders import GraphBuilder
from parsers.semantic.model import TextConcepts
from utils.commons import safe_concurrency_backend, ModuleShutUpWarning

__all__ = ['compute_graph', 'compute_graph_from_concepts_file', 'compute_graphs', 'compute_graphs_from_concepts_dir',
           'concepts_dir_to_graph_files']

LOG = logging.getLogger(__name__)


def compute_graph(text_concepts: TextConcepts, graph_builder: GraphBuilder, out_filename: str):
    graph = graph_builder.build_graph_from_text_concepts(text_concepts)
    if out_filename is not None:
        LOG.debug("Save the graph %s" % out_filename)
        graph_builder.to_json(out_filename, graph)
    return graph


def compute_graph_from_concepts_file(filename: str, graph_builder: GraphBuilder, out_filename: str):
    with open(filename, 'r') as f_in:
        text_concepts = TextConcepts.from_dict(json.load(f_in))
    return compute_graph(text_concepts, graph_builder, out_filename)


def compute_graphs(texts_concepts: Iterable[TextConcepts], graph_builder: GraphBuilder, n_jobs: int,
                   backend: str = 'multiprocessing', out_filenames: Iterable[str] = None) -> Iterable[Any]:
    backend = safe_concurrency_backend(backend, heavy_sharing=True)

    if out_filenames is None:
        out_filenames = []
    gen = zip_longest(texts_concepts, out_filenames)
    with ModuleShutUpWarning('rdflib'):
        return Parallel(n_jobs=n_jobs, verbose=5, backend=backend)(
            delayed(compute_graph)(tc, graph_builder, of) for tc, of in gen
        )


def compute_graphs_from_concepts_dir(dir_in: str, graph_builder: GraphBuilder, n_jobs: int,
                                     backend: str = 'multiprocessing', dir_out: str = None,
                                     in_ext: str = ".json", out_ext: str = ".json"):
    backend = safe_concurrency_backend(backend, heavy_sharing=True)

    gen = glob.glob(os.path.join(dir_in, '*' + in_ext))
    if dir_out:
        gen = ((file_in, os.path.join(dir_out, os.path.splitext(os.path.basename(file_in))[0] + out_ext))
               for file_in in gen)
    else:
        gen = ((f, None) for f in gen)

    with ModuleShutUpWarning('rdflib'):
        return Parallel(n_jobs=n_jobs, verbose=5, backend=backend)(
            delayed(compute_graph_from_concepts_file)(f_in, graph_builder, f_out) for f_in, f_out in gen
        )


def concepts_dir_to_graph_files(dir_in: str, dir_out: str, graph_builder: GraphBuilder, n_jobs: int,
                                backend: str = 'multiprocessing', force_rewrite: bool = False,
                                in_ext: str = ".json", out_ext: str = ".json"):
    backend = safe_concurrency_backend(backend, heavy_sharing=True)

    gen = glob.glob(os.path.join(dir_in, '*' + in_ext))
    if dir_out:
        gen = ((file_in, os.path.join(dir_out, os.path.splitext(os.path.basename(file_in))[0] + out_ext))
               for file_in in gen)
        if not force_rewrite:
            gen = filter(lambda x: not os.path.exists(x[1]), gen)
    else:
        gen = ((f, None) for f in gen)

    with ModuleShutUpWarning('rdflib'):
        Parallel(n_jobs=n_jobs, verbose=5, backend=backend)(
            delayed(compute_graph_from_concepts_file)(f_in, graph_builder, f_out) for f_in, f_out in gen
        )
