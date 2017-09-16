import pprint
import configparser

pp = pprint.PrettyPrinter()

class Config(object):
    def __init__(self, paths_config, verbose=False):
        config = configparser.ConfigParser()
        config._interpolation = configparser.ExtendedInterpolation()
        config.read(paths_config)
        print(paths_config)

        c = config['DEFAULT']

        d = {}
        for k in c:
            d[k] = c[k]

        self.resources_dir = d['resources_dir']

        self.vocab_dir = d['vocab_dir']

        # Word2Vec Vocab to Idxs
        self.word_vocab_pkl = d['word_vocab_pkl']
        # Wid2Idx for Known Entities ~ 620K (readers.train.vocab.py)
        self.kwnwid_vocab_pkl = d['kwnwid_vocab_pkl']
        # FIGER Type label 2 idx (readers.train.vocab.py)
        self.label_vocab_pkl = d['label_vocab_pkl']
        # EntityWid: [FIGER Type Labels]
        # CoherenceStr2Idx at various thresholds (readers.train.vocab.py)
        self.cohstringG9_vocab_pkl = d['cohstringg9_vocab_pkl']

        # wid2Wikititle for whole KB ~ 3.18M (readers.train.vocab.py)
        self.widWiktitle_pkl = d['widwiktitle_pkl']

        self.crosswikis_pruned_pkl = d['crosswikis_pruned_pkl']

        self.glove_pkl = d['glove_pkl']
        self.glove_word_vocab_pkl = d['glove_word_vocab_pkl']

        self.test_file = d['test_file']

        if verbose:
            pp.pprint(d)

    #endinit

if __name__=='__main__':
    c = Config("configs/allnew_mentions_config.ini", verbose=True)
