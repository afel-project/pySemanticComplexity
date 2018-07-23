# -*- coding: utf-8 -*-
import re

import numpy as np
from sklearn.base import BaseEstimator

from utils.stanfordResources import TRegexCounter, LexParser

__all__ = ['StanfordSyntacticTransformer']


class StanfordSyntacticTransformer(BaseEstimator):
    __FEATURES = ["W", "S", "VP", "C", "T", "DC", "CT", "CP", "CN", "MLS", "MLT", "MLC", "C/S", "VP/T", "C/T", "DC/C",
                  "DC/T", "T/S", "CT/T", "CP/T", "CP/C", "CN/T", "CN/C"]

    _PATTERNS_LIST = [
        "ROOT",  # sentence (S)
        "VP > S|SINV|SQ",  # verb phrase (VP)
        "S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])]",
        # clause (C)
        "S|SBARQ|SINV|SQ > ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP]",  # T-unit (T)
        "SBAR < (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])])",
        # dependent clause (DC)
        "S|SBARQ|SINV|SQ [> ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP]] << (SBAR < (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])]))",
        # complex T-unit (CT)
        "ADJP|ADVP|NP|VP < CC",  # coordinate phrase (CP)
        "NP !> NP [<< JJ|POS|PP|S|VBG | << (NP $++ NP !$+ CC)]",  # complex nominal (CN1)
        "SBAR [<# WHNP | <# (IN < That|that|For|for) | <, S] & [$+ VP | > VP]",  # complex nominal (CN2)
        "S < (VP <# VBG|TO) $+ VP",  # complex nominal (CN3)
        "FRAG > ROOT !<< (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])])",
        # fragment clause
        "FRAG > ROOT !<< (S|SBARQ|SINV|SQ > ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP])",  # fragment T-unit
        "MD|VBZ|VBP|VBD > (SQ !< VP)",  # verb phrase (VP2)
    ]

    _WORD_COUNT_RE = re.compile("\([A-Z]+\$? [^)(]+\)")

    def __init__(self, file_names: bool = False):
        self.file_names = file_names

    def fit_transform(self, X, y=None):
        processing = np.vectorize(lambda x: self.compute_features(x),
                                  signature='()->(n)', otypes=[float])
        return processing(X)

    @classmethod
    def get_features(cls):
        return cls.__FEATURES

    def compute_features(self, text):
        # Parse the text to Tree
        lex_parser = LexParser()
        parsed_text = lex_parser.parse(text, is_file_name=self.file_names)
        # Compute the number of words
        words_count = self._compute_words_count(parsed_text)
        # Compute the patterns
        patterns_count = self._compute_patterns_count(parsed_text)
        # Compute the complexity features
        complexity_features = self._compute_complexity_features(patterns_count, words_count)
        # return merged array of patterns count and complexity features
        return np.hstack([words_count, patterns_count, complexity_features])

    def _compute_words_count(self, text):
        return len(self._WORD_COUNT_RE.findall(text))

    def _compute_patterns_count(self, parsed_text):
        tregex = TRegexCounter()
        temp_file = tregex.create_temporary_input_file(parsed_text)
        try:
            # Call the tree regex standford parser of the parsed file of each regex
            patterncount = np.vectorize(lambda x: tregex.count(x, temp_file.name, is_file_name=True),
                                        signature='()->()', otypes=[int])(self._PATTERNS_LIST)
            # update frequencies of complex nominals, clauses, and T-units
            patterncount[7] = patterncount[-4] + patterncount[-5] + patterncount[-6]
            patterncount[2] = patterncount[2] + patterncount[-3]
            patterncount[3] = patterncount[3] + patterncount[-2]
            patterncount[1] = patterncount[1] + patterncount[-1]
            return patterncount[:8]
        finally:
            temp_file.close()

    @staticmethod
    def _compute_complexity_features(patterns_count, words_count):
        def division(x, y):
            if float(x) == 0 or float(y) == 0:
                return 0
            return float(x) / float(y)

        # list of frequencies of structures other than words
        [s, vp, c, t, dc, ct, cp, cn] = patterns_count
        # compute the 14 syntactic complexity indices
        mls = division(words_count, s)
        mlt = division(words_count, t)
        mlc = division(words_count, c)
        c_s = division(c, s)
        vp_t = division(vp, t)
        c_t = division(c, t)
        dc_c = division(dc, c)
        dc_t = division(dc, t)
        t_s = division(t, s)
        ct_t = division(ct, t)
        cp_t = division(cp, t)
        cp_c = division(cp, c)
        cn_t = division(cn, t)
        cn_c = division(cn, c)
        return np.array([mls, mlt, mlc, c_s, vp_t, c_t, dc_c, dc_t, t_s, ct_t, cp_t, cp_c, cn_t, cn_c])
