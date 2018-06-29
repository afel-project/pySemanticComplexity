# -*- coding: utf-8 -*-
"""Define the main entities the program deals with to express entities from DBPedia."""

__all__ = ['AnnotationScore', 'DBpediaResource']


class AnnotationScore:
    __slots__ = ['offset', 'surface', 'similarity_score', 'percentage_second_rank', 'support']

    def __init__(self, offset: int, surface_form: str, similarity_score: float = None,
                 percentage_second_rank: float = None,
                 support: int = None):
        self.offset = offset
        self.surface = surface_form
        self.similarity_score = similarity_score
        self.percentage_second_rank = percentage_second_rank
        self.support = support

    def __repr__(self):
        o = self.offset if self.offset is not None else -1
        sf = self.surface if self.surface is not None else 'none'
        ss = self.similarity_score if self.similarity_score is not None else -1.0
        psr = self.percentage_second_rank if self.percentage_second_rank is not None else -1.0
        su = self.support if self.support is not None else -1
        return "<offset: %d, surface: %s, similarity_score: %.3f, secRank: %.3f, support: %d>" % (o, sf, ss, psr, su)

    def to_dict(self):
        attrs = (('offset', self.offset), ('surfaceForm', self.surface), ('similarityScore', self.similarity_score),
                 ('percentageSecondRank', self.percentage_second_rank), ('support', self.support))
        return dict(filter(lambda x: x[1] is not None, attrs))

    @classmethod
    def from_dict(cls, data):
        return cls(offset=data.get('offset'), surface_form=data.get('surfaceForm'),
                   similarity_score=data.get('similarityScore'),
                   percentage_second_rank=data.get('percentageSecondRank'), support=data.get('support'))


class DBpediaResource:
    __slots__ = ['uri', 'types']

    def __init__(self, uri: str, types: [str] = None):
        self.uri = uri
        self.types = types if types is not None else []

    def __repr__(self):
        return "%s %s" % (self.uri, str(self.types))

    def to_dict(self):
        attrs = (('uri', self.uri), ('types', self.types))
        return dict(filter(lambda x: x[1] is not None, attrs))

    @classmethod
    def from_dict(cls, data):
        return cls(uri=data.get('uri'), types=data.get('types', []))
