# -*- coding: utf-8 -*-
"""Simple tool to draw a graph from a graph json file."""
import re
from argparse import Namespace, ArgumentParser
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx

from parsers.semantic.graphs.builders import NetworkXGraphBuilder
from utils.commons import BatchProcess

__all__ = ['PrintGraph']


class PrintGraph(BatchProcess):
    _NAMESPACES = {
        'DBPedia': re.compile('^http://dbpedia.org/ontology/'),
        'Schema': re.compile('^http://schema.org/'),
        'yago': re.compile('^http://dbpedia.org/class/yago/'),
        'ROOT': re.compile('^#AbstractConcept#')
    }

    _NS_COLORS = {
        'DBPedia': 'red',
        'Schema': 'blue',
        'yago': 'purple',
        'ROOT': 'black',
    }
    _DFLT_COLOR = 'green'

    def __init__(self, *largs, **kwargs):
        kwargs['use_logger'] = False
        super().__init__(*largs, **kwargs)

    def _configure_args(self, parser: ArgumentParser) -> ArgumentParser:
        parser.description = "Draw a concepts graph using pyplot engine. Require Matplotlib to be installed."
        parser.add_argument('graph_file', help='Graphs file', metavar='<Graph>', type=str)
        return parser

    def _run(self, args: Namespace) -> Optional[int]:
        graph = self._load_graph(args.graph_file)
        self._draw_graph(graph)
        input('Press enter to continue:')
        return

    @classmethod
    def _load_graph(cls, filename: str):
        return NetworkXGraphBuilder.from_json(filename)

    @classmethod
    def _draw_graph(cls, graph: nx.Graph):
        names = cls._create_qnames(graph)
        colors = cls._create_colors(graph, names)

        f = plt.figure(figsize=(14, 7))
        # draw_networkx
        # nx.draw_networkx(graph,
        #                  node_size=100, node_color=colors,
        #                  with_labels=True, labels=names, font_size=8,
        #                  edge_color='grey'
        #                  )
        nx.draw_networkx(graph,
                         node_size=100, node_color=colors,
                         with_labels=False,
                         edge_color='grey'
                         )
        f.show()

    @classmethod
    def _create_qnames(cls, graph: nx.Graph):
        return dict((uri, cls._extract_qname(uri)) for uri in graph.nodes)

    @classmethod
    def _create_colors(cls, graph: nx.Graph, qnames: dict):
        return [cls._NS_COLORS.get(qnames[node].split(':', 1)[0], cls._DFLT_COLOR) for node in graph.nodes]

    @classmethod
    def _extract_qname(cls, uri: str):
        candidates = ((key, r.sub('', uri, 1)) for key, r in cls._NAMESPACES.items() if r.match(uri))
        candidates = ((key, suffix, len(suffix)) for key, suffix in candidates)
        candidates = sorted(candidates, key=lambda x: x[2], reverse=False)
        if candidates:
            key, suffix, _ = candidates[0]
            if suffix:
                return key + ':' + suffix
                # return suffix
            else:
                return key
        else:
            return uri.split('/')[-1]


if __name__ == '__main__':
    PrintGraph().start()
