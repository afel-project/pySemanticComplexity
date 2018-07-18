# -*- coding: utf-8 -*-
"""Define the main entities the program deals with to express entities from DBPedia."""
from typing import List, Iterable

__all__ = ['AnnotationScore', 'DBpediaResource', 'TextConcepts', 'ConceptInformation']


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
                   percentage_second_rank=data.get('percentageSecondRank'), support=data.get('support')) \
            if data is not None else None


class DBpediaResource:
    __slots__ = ['uri', 'types', 'scores']

    def __init__(self, uri: str, types: [str] = None, scores: AnnotationScore = None):
        self.uri = uri
        self.types = types if types is not None else []
        self.scores = scores

    def __repr__(self):
        return "%s %s" % (self.uri, str(self.types))

    def __eq__(self, other):
        if isinstance(other, DBpediaResource):
            return self.uri == other.uri
        else:
            return NotImplemented

    def __hash__(self):
        return hash(self.uri)

    def to_dict(self):
        attrs = (('uri', self.uri), ('types', self.types), ('scores', self.scores.to_dict()))
        return dict(filter(lambda x: x[1] is not None, attrs))

    @classmethod
    def from_dict(cls, data):
        return cls(uri=data.get('uri'), types=data.get('types', []),
                   scores=AnnotationScore.from_dict(data.get('scores'))) if data is not None else None


class TextConcepts:
    __slots__ = ['concepts', 'nb_words']

    def __init__(self, concepts: List[DBpediaResource], nb_words: int):
        self.concepts = concepts
        self.nb_words = nb_words

    def to_dict(self):
        return {'concepts': [c.to_dict() for c in self.concepts],
                'nbWords': self.nb_words}

    @classmethod
    def from_dict(cls, data):
        concepts = [DBpediaResource.from_dict(d) for d in data.get('concepts', [])]
        nb_words = data.get('nbWords')
        return TextConcepts(concepts, nb_words)


class ConceptInformation:
    __slots__ = ['concept_iri', 'types', 'nb_links_in', 'nb_links_out']

    def __init__(self, concept_iri: str, types: Iterable[str], nb_links_in: int, nb_links_out: int):
        self.concept_iri = concept_iri
        self.types = types
        self.nb_links_in = nb_links_in
        self.nb_links_out = nb_links_out

    def to_dict(self):
        return dict((k, getattr(self, k)) for k in self.__slots__)

    @classmethod
    def from_dict(cls, data):
        return ConceptInformation(**data)

    @classmethod
    def load_concept_information_dict_from_json(cls, data):
        return dict((k, cls.from_dict(v)) for k, v in data.items())
