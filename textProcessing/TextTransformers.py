# -*- coding: utf-8 -*-
from typing import List

import numpy as np
from nltk.tokenize.treebank import TreebankWordTokenizer
from sklearn.base import BaseEstimator


class TextTransformer(BaseEstimator):
    def __init__(self):
        self._word_tokenizer = TreebankWordTokenizer()

    @property
    def word_tokenizer(self):
        return self._word_tokenizer

    def tokenize(self, text: str) -> List[str]:
        return self._word_tokenizer.tokenize(text)

    def count_words(self, text: str) -> int:
        return len(self.tokenize(text))

    def fit_transform(self, X, y=None):
        return np.vectorize(self.count_words, signature='(n)->(n)')(X)

    def get_features_names(self) -> [str]:
        return ["nbWords"]
