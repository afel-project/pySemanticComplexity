# -*- coding: utf-8 -*-
"""Text precessor to clean raw text or split it in paragraphs or sentences."""
import re
from typing import Any, List

from nltk.tokenize.treebank import TreebankWordTokenizer

__all__ = ['TextPreprocessor']


class TextPreprocessor:
    _WRONG_CHAR_FILTER = re.compile(
        '[' + ''.join([chr(i) for i in range(0, 0x0a)]) + ''.join([chr(i) for i in range(0x0b, 0x20)]) + ''.join(
            [chr(i) for i in range(0x80, 0x9f)]) + ']')
    _PARAGRAPH_FILTER = re.compile('\n\n')

    def __init__(self, sentence_tokenizer: Any = None, paragraph_threshold: int = 150):
        """
        Constructor
        :param sentence_tokenizer: a sentences_tokenizer that provide a tokenize(t:str)->[str] method
        (for instance: nltk.data.load('tokenizers/punkt/english.pickle'))
        :param paragraph_threshold: the minimum number of characters of paragraph should contains (it will be
        filtered otherwise)
        """
        self.sentences_tokenizer = sentence_tokenizer  # might wanna use
        self.paragraph_threshold = paragraph_threshold
        self._word_tokenizer = TreebankWordTokenizer()

    def clean_text(self, text: str) -> str:
        return self._WRONG_CHAR_FILTER.sub(" ", text)

    def split_to_paragraphs(self, text: str) -> List[str]:
        return re.split(self._PARAGRAPH_FILTER, text)

    def filter_paragraphs(self, paragraphs: List[str]) -> List[str]:
        return list(filter(lambda x: len(x) > self.paragraph_threshold, paragraphs))

    '''
    def split_to_sentences(self, text: str) -> Iterable[str]:
        if self.sentences_tokenizer is None:
            raise AttributeError("No tokenizer has been set")

        return self.sentences_tokenizer.tokenize(text)
    '''

    def process_to_paragraphs(self, text: str) -> List[str]:
        r = text

        r = self.clean_text(r)
        r = self.split_to_paragraphs(r)
        r = self.filter_paragraphs(r)

        return r

    @property
    def word_tokenizer(self):
        return self._word_tokenizer

    def tokenize(self, text: str) -> List[str]:
        return self._word_tokenizer.tokenize(text)

    def count_words(self, text: str) -> int:
        return len(self.tokenize(text))
