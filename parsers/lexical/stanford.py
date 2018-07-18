# -*- coding: utf-8 -*-
import logging
import os
import random
import string
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator

from utils.resources import VENDOR_DIR_PATH
from utils.stanfordResources import PosTagger

LOG = logging.getLogger(__name__)

__all__ = ['BNCStanfordLexicalTransformer', 'ANCStanfordLexicalTransformer', 'StanfordLexicalTransformer']


class WordListInformationManager(metaclass=ABCMeta):
    def __init__(self):
        self.worddict = {}
        self.adjdict = {}
        self.verbdict = {}
        self.noundict = {}
        self.load_file()
        self.wordranks = self._sort_key_by_value(self.worddict)
        self.verbranks = self._sort_key_by_value(self.verbdict)

    @abstractmethod
    def load_file(self):
        pass

    @staticmethod
    def _sort_key_by_value(d):
        return [value for _, value in sorted(((v, k) for k, v in d.items()))]


class BNCWordListInformationManager(WordListInformationManager):
    __FILEPATH = os.path.join(VENDOR_DIR_PATH, "stanford/lca/bnc_all_filtered.txt")

    def __init__(self):
        super().__init__()

    def load_file(self):
        with open(self.__FILEPATH, 'r') as fin:
            for word_info in (l.strip() for l in fin):
                if not word_info or "Total words" in word_info:
                    continue
                [lemma, pos, frequency] = word_info.split()[:3]
                frequency = int(frequency)
                self.worddict[lemma] = self.worddict.get(lemma, 0) + frequency
                if pos == "Adj":
                    self.adjdict[lemma] = self.adjdict.get(lemma, 0) + frequency
                elif pos == "Verb":
                    self.verbdict[lemma] = self.verbdict.get(lemma, 0) + frequency
                elif pos == "NoC" or pos == "NoP":
                    self.noundict[lemma] = self.noundict.get(lemma, 0) + frequency


class ANCWordListInformationManager(WordListInformationManager):
    __FILEPATH = os.path.join(VENDOR_DIR_PATH, "stanford/lca/anc_all_count.txt")

    def __init__(self):
        super().__init__()

    def load_file(self):
        with open(self.__FILEPATH, 'r') as fin:
            for word_info in (l.strip() for l in fin):
                if not word_info or "Total words" in word_info:
                    continue
                [_, lemma, pos, frequency] = word_info.split()[:4]
                frequency = int(frequency)
                self.worddict[lemma] = self.worddict.get(lemma, 0) + frequency
                if pos[0] == "J":
                    self.adjdict[lemma] = self.adjdict.get(lemma, 0) + frequency
                elif pos[0] == "V":
                    self.verbdict[lemma] = self.verbdict.get(lemma, 0) + frequency
                elif pos[0] == "N":
                    self.noundict[lemma] = self.noundict.get(lemma, 0) + frequency


class StanfordLexicalTransformer(BaseEstimator, metaclass=ABCMeta):
    __FEATURES = ["sentences", "wordtypes", "swordtypes", "lextypes", "slextypes", "wordtokens",
                  "swordtokens", "lextokens", "slextokens", "ld", "ls1", "ls2", "vs1", "vs2", "cvs1", "ndw", "ndwz",
                  "ndwerz", "ndwesz", "ttr", "msttr", "cttr", "rttr", "logttr", "uber", "lv", "vv1", "svv1", "cvv1",
                  "vv2", "nv", "adjv", "advv", "modv"]

    def __init__(self, word_rank_limit: int = 2000, sample_size_mini: int = 50, pos_tagger=None,
                 file_names: bool = False):
        self.word_rank_limit = word_rank_limit
        self.sample_size_mini = sample_size_mini
        self.pos_tagger = pos_tagger
        if self.pos_tagger is None:
            self.pos_tagger = PosTagger()
        self.file_names = file_names

    def fit_transform(self, X, y=None):
        processing = np.vectorize(lambda x: self.compute_features(x),
                                  signature='()->(n)', otypes=[float])
        return processing(X)

    def compute_features(self, text):
        return self.pos_tags_to_complexity(self.text_to_pos_tags(text))

    @classmethod
    def get_features(cls):
        return cls.__FEATURES

    @abstractmethod
    def get_word_list_information_manager(self) -> WordListInformationManager:
        pass

    @classmethod
    def _is_letter_number(cls, character: str) -> int:
        if character in string.printable and character not in string.punctuation:
            return 1
        return 0

    @classmethod
    def _is_sentence(cls, line: str) -> int:
        for character in line:
            if cls._is_letter_number(character):
                return 1
        return 0

    # NDW for first z words in a sample
    @staticmethod
    def _getndwfirstz(z, lemmalist):
        ndwfirstztype = {}
        for lemma in lemmalist[:z]:
            ndwfirstztype[lemma] = 1
        return len(ndwfirstztype.keys())

    # NDW expected random z words, 10 trials
    @staticmethod
    def _getndwerz(z, lemmalist):
        ndwerz = 0
        for i in range(10):
            ndwerztype = {}
            erzlemmalist = random.sample(lemmalist, z)
            for lemma in erzlemmalist:
                ndwerztype[lemma] = 1
            ndwerz += len(ndwerztype.keys())
        return ndwerz / 10.0

    # NDW expected random sequences of z words, 10 trials
    @staticmethod
    def _getndwesz(z, lemmalist):
        ndwesz = 0
        for i in range(10):
            ndwesztype = {}
            startword = random.randint(0, len(lemmalist) - z)
            eszlemmalist = lemmalist[startword:startword + z]
            for lemma in eszlemmalist:
                ndwesztype[lemma] = 1
            ndwesz += len(ndwesztype.keys())
        return ndwesz / 10.0

    # MSTTR
    @staticmethod
    def _getmsttr(z, lemmalist):
        samples = 0
        msttr = 0.0
        while len(lemmalist) >= z:
            samples += 1
            msttrtype = {}
            for lemma in lemmalist[:z]:
                msttrtype[lemma] = 1
            msttr += len(msttrtype.keys()) / float(z)
            lemmalist = lemmalist[z:]
        return msttr / samples

    def text_to_pos_tags(self, text):
        return self.pos_tagger.tag_pos(text, is_file_name=self.file_names)

    def pos_tags_to_complexity(self, line_generator):
        wi_mgr = self.get_word_list_information_manager()
        wordtypes = {}
        wordtokens = 0
        swordtypes = {}
        swordtokens = 0
        lextypes = {}
        lextokens = 0
        slextypes = {}
        slextokens = 0
        verbtypes = {}
        verbtokens = 0
        sverbtypes = {}
        adjtypes = {}
        adjtokens = 0
        advtypes = {}
        advtokens = 0
        nountypes = {}
        nountokens = 0
        sentences = 0
        lemmaposlist = []
        lemmalist = []

        for lemline in line_generator:
            lemline = lemline.strip()
            lemline = lemline.lower()
            if not self._is_sentence(lemline):
                continue
            sentences += 1
            lemmas = lemline.split()
            for lemma in lemmas:
                word = lemma.split("_")[0]
                pos = lemma.split("_")[-1]
                if (pos not in string.punctuation) and pos != "sent" and pos != "sym":
                    lemmaposlist.append(lemma)
                    lemmalist.append(word)
                    wordtokens += 1
                    wordtypes[word] = 1
                    if (word not in wi_mgr.wordranks[-self.word_rank_limit:]) and pos != "cd":
                        swordtypes[word] = 1
                        swordtokens += 1
                    if pos[0] == "n":
                        lextypes[word] = 1
                        nountypes[word] = 1
                        lextokens += 1
                        nountokens += 1
                        if word not in wi_mgr.wordranks[-self.word_rank_limit:]:
                            slextypes[word] = 1
                            slextokens += 1
                    elif pos[0] == "j":
                        lextypes[word] = 1
                        adjtypes[word] = 1
                        lextokens += 1
                        adjtokens += 1
                        if word not in wi_mgr.wordranks[-self.word_rank_limit:]:
                            slextypes[word] = 1
                            slextokens += 1
                    elif pos[0] == "r" and (
                            word in wi_mgr.adjdict or (word[-2:] == "ly" and word[:-2] in wi_mgr.adjdict)):
                        lextypes[word] = 1
                        advtypes[word] = 1
                        lextokens += 1
                        advtokens += 1
                        if word not in wi_mgr.wordranks[-self.word_rank_limit:]:
                            slextypes[word] = 1
                            slextokens += 1
                    elif pos[0] == "v" and word not in ["be", "have"]:
                        verbtypes[word] = 1
                        verbtokens += 1
                        lextypes[word] = 1
                        lextokens += 1
                        if word not in wi_mgr.wordranks[-self.word_rank_limit:]:
                            sverbtypes[word] = 1
                            slextypes[word] = 1
                            slextokens += 1

        # 0. basic statistics
        mls = wordtokens / float(sentences)

        # 1. lexical density
        ld = float(lextokens) / wordtokens

        # 2. lexical sophistication
        # 2.1 lexical sophistication
        ls1 = slextokens / float(lextokens)
        ls2 = len(swordtypes.keys()) / float(len(wordtypes.keys()))

        # 2.2 verb sophistication
        vs1 = len(sverbtypes.keys()) / float(verbtokens)
        vs2 = (len(sverbtypes.keys()) * len(sverbtypes.keys())) / float(verbtokens)
        cvs1 = len(sverbtypes.keys()) / np.sqrt(2 * verbtokens)

        # 3 lexical diversity or variation

        # 3.1 NDW, may adjust the values of "standard"
        ndw = ndwz = ndwerz = ndwesz = len(wordtypes.keys())
        if len(lemmalist) >= self.sample_size_mini:
            ndwz = self._getndwfirstz(self.sample_size_mini, lemmalist)
            ndwerz = self._getndwerz(self.sample_size_mini, lemmalist)
            ndwesz = self._getndwesz(self.sample_size_mini, lemmalist)

        # 3.2 TTR
        msttr = ttr = len(wordtypes.keys()) / float(wordtokens)
        if len(lemmalist) >= self.sample_size_mini:
            msttr = self._getmsttr(self.sample_size_mini, lemmalist)
        cttr = len(wordtypes.keys()) / np.sqrt(2 * wordtokens)
        rttr = len(wordtypes.keys()) / np.sqrt(wordtokens)
        logttr = np.log(len(wordtypes.keys())) / np.log(wordtokens)
        uber = (np.log10(wordtokens) * np.log10(wordtokens)) / np.log10(wordtokens / float(len(wordtypes.keys())))

        # 3.3 verb diversity
        vv1 = len(verbtypes.keys()) / float(verbtokens)
        svv1 = len(verbtypes.keys()) * len(verbtypes.keys()) / float(verbtokens)
        cvv1 = len(verbtypes.keys()) / np.sqrt(2 * verbtokens)

        # 3.4 lexical diversity
        lv = len(lextypes.keys()) / float(lextokens)
        vv2 = len(verbtypes.keys()) / float(lextokens)
        nv = len(nountypes.keys()) / float(nountokens)
        adjv = len(adjtypes.keys()) / float(lextokens)
        advv = len(advtypes.keys()) / float(lextokens)
        modv = (len(advtypes.keys()) + len(adjtypes.keys())) / float(lextokens)

        return np.array([sentences, len(wordtypes.keys()), len(swordtypes.keys()), len(lextypes.keys()),
                         len(slextypes.keys()), wordtokens, swordtokens, lextokens, slextokens, ld, ls1, ls2, vs1,
                         vs2, cvs1, ndw, ndwz, ndwerz, ndwesz, ttr, msttr, cttr, rttr, logttr, uber, lv, vv1, svv1,
                         cvv1, vv2, nv, adjv, advv, modv])


class BNCStanfordLexicalTransformer(StanfordLexicalTransformer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._words_mgr = BNCWordListInformationManager()

    def get_word_list_information_manager(self):
        return self._words_mgr


class ANCStanfordLexicalTransformer(StanfordLexicalTransformer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._words_mgr = ANCWordListInformationManager()

    def get_word_list_information_manager(self):
        return self._words_mgr
