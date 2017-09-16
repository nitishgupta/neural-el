import re
import os
import gc
import sys
import time
import math
import pickle
import random
import gensim
import pprint
import unicodedata
import configparser
import collections
import numpy as np
import readers.utils as utils
from readers.config import Config

class VocabLoader(object):
    def __init__(self, config):
        self.initialize_all_dicts()
        self.config = config

    def initialize_all_dicts(self):
        (self.word2idx, self.idx2word) = (None, None)
        (self.label2idx, self.idx2label) = (None, None)
        (self.knwid2idx, self.idx2knwid) = (None, None)
        self.wid2Wikititle = None
        self.wid2TypeLabels = None
        (self.test_knwen_cwikis, self.test_allen_cwikis) = (None, None)
        self.cwikis_slice = None
        self.glove2vec = None
        (self.gword2idx, self.gidx2word) = (None, None)
        self.crosswikis = None

    def loadCrosswikis(self):
        if self.crosswikis == None:
            if not os.path.exists(self.config.crosswikis_pkl):
                print("Crosswikis pkl missing")
                sys.exit()
            self.crosswikis = utils.load(self.config.crosswikis_pkl)
        return self.crosswikis

    def getWordVocab(self):
        if self.word2idx == None or self.idx2word == None:
            if not os.path.exists(self.config.word_vocab_pkl):
                print("Word Vocab PKL missing")
                sys.exit()
            print("Loading Word Vocabulary")
            (self.word2idx, self.idx2word) = utils.load(self.config.word_vocab_pkl)
        return (self.word2idx, self.idx2word)

    def getLabelVocab(self):
        if self.label2idx == None or self.idx2label == None:
            if not os.path.exists(self.config.label_vocab_pkl):
                print("Label Vocab PKL missing")
                sys.exit()
            print("Loading Type Label Vocabulary")
            (self.label2idx, self.idx2label) = utils.load(self.config.label_vocab_pkl)
        return (self.label2idx, self.idx2label)

    def getKnwnWidVocab(self):
        if self.knwid2idx == None or self.idx2knwid == None:
            if not os.path.exists(self.config.kwnwid_vocab_pkl):
                print("Known Entities Vocab PKL missing")
                sys.exit()
            print("Loading Known Entity Vocabulary ... ")
            (self.knwid2idx, self.idx2knwid) = utils.load(self.config.kwnwid_vocab_pkl)
        return (self.knwid2idx, self.idx2knwid)

    def getTestKnwEnCwiki(self):
        if self.test_knwen_cwikis == None:
            if not os.path.exists(self.config.test_kwnen_cwikis_pkl):
                print("Test Known Entity CWikis Dict missing")
                sys.exit()
            print("Loading Test Data Known Entity CWIKI")
            self.test_knwen_cwikis = utils.load(self.config.test_kwnen_cwikis_pkl)
        return self.test_knwen_cwikis

    def getTestAllEnCwiki(self):
        if self.test_allen_cwikis == None:
            if not os.path.exists(self.config.test_allen_cwikis_pkl):
                print("Test All Entity CWikis Dict missing")
                sys.exit()
            print("Loading Test Data All Entity CWIKI")
            self.test_allen_cwikis = utils.load(self.config.test_allen_cwikis_pkl)
        return self.test_allen_cwikis

    def getCrosswikisSlice(self):
        if self.cwikis_slice == None:
            if not os.path.exists(self.config.crosswikis_slice):
                print("CWikis Slice Dict missing")
                sys.exit()
            print("Loading CWIKI Slice")
            self.cwikis_slice = utils.load(self.config.crosswikis_slice)
        return self.cwikis_slice

    def getWID2Wikititle(self):
        if self.wid2Wikititle == None:
            if not os.path.exists(self.config.widWiktitle_pkl):
                print("wid2Wikititle pkl missing")
                sys.exit()
            print("Loading wid2Wikititle")
            self.wid2Wikititle = utils.load(self.config.widWiktitle_pkl)
        return self.wid2Wikititle

    def getWID2TypeLabels(self):
        if self.wid2TypeLabels == None:
            if not os.path.exists(self.config.wid2typelabels_vocab_pkl):
                print("wid2TypeLabels pkl missing")
                sys.exit()
            print("Loading wid2TypeLabels")
            self.wid2TypeLabels = utils.load(self.config.wid2typelabels_vocab_pkl)
        return self.wid2TypeLabels

    def loadGloveVectors(self):
        if self.glove2vec == None:
            if not os.path.exists(self.config.glove_pkl):
                print("Glove_Vectors_PKL doesnot exist")
                sys.exit()
            print("Loading Glove Word Vectors")
            self.glove2vec = utils.load(self.config.glove_pkl)
        return self.glove2vec

    def getGloveWordVocab(self):
        if self.gword2idx == None or self.gidx2word == None:
            if not os.path.exists(self.config.glove_word_vocab_pkl):
                print("Glove Word Vocab PKL missing")
                sys.exit()
            print("Loading Glove Word Vocabulary")
            (self.gword2idx, self.gidx2word) = utils.load(self.config.glove_word_vocab_pkl)
        return (self.gword2idx, self.gidx2word)

if __name__=='__main__':
    config = Config("configs/wcoh_config.ini")
    a = VocabLoader(config)
    a.loadWord2Vec()
