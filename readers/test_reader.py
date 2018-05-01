import time
import numpy as np
import readers.utils as utils
from readers.Mention import Mention
from readers.config import Config
from readers.vocabloader import VocabLoader

start_word = "<s>"
end_word = "<eos>"

class TestDataReader(object):
    def __init__(self, config, vocabloader, test_mens_file,
                 num_cands, batch_size, strict_context=True,
                 pretrain_wordembed=True, coherence=True,
                 glove=True):
        print("Loading Test Reader: {}".format(test_mens_file))
        self.typeOfReader="test"
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = 'unk'  # In tune with word2vec
        self.unk_wid = "<unk_wid>"
        # self.useKnownEntitesOnly = True
        self.pretrain_wordembed = pretrain_wordembed
        self.coherence = coherence

        # Word Vocab
        (self.word2idx, self.idx2word) = vocabloader.getGloveWordVocab()
        self.num_words = len(self.idx2word)
        print(" [#] Word vocab loaded. Size of vocab : {}".format(
            self.num_words))

        # Label Vocab
        (self.label2idx, self.idx2label) = vocabloader.getLabelVocab()
        self.num_labels = len(self.idx2label)
        print(" [#] Label vocab loaded. Number of labels : {}".format(
            self.num_labels))

        # Known WID Vocab
        (self.knwid2idx, self.idx2knwid) = vocabloader.getKnwnWidVocab()
        self.num_knwn_entities = len(self.idx2knwid)
        print(" [#] Loaded. Num of known wids : {}".format(
            self.num_knwn_entities))

        # Wid2Wikititle Map
        self.wid2WikiTitle = vocabloader.getWID2Wikititle()
        print(" [#] Size of Wid2Wikititle: {}".format(len(
            self.wid2WikiTitle)))

        # # Wid2TypeLabels Map
        # self.wid2TypeLabels = vocabloader.getWID2TypeLabels()
        # print(" [#] Total number of Wids : {}".format(len(
        #     self.wid2TypeLabels)))

        # Coherence String Vocab
        print("Loading Coherence Strings Dicts ... ")
        (self.cohG92idx, self.idx2cohG9) = utils.load(
            config.cohstringG9_vocab_pkl)
        self.num_cohstr = len(self.idx2cohG9)
        print(" [#] Number of Coherence Strings in Vocab : {}".format(
            self.num_cohstr))

        # Known WID Description Vectors
        # self.kwnwid2descvecs = vocabloader.loadKnownWIDDescVecs()
        # print(" [#] Size of kwn wid desc vecs dict : {}".format(
        #     len(self.kwnwid2descvecs)))

        # # Crosswikis
        # print("[#] Loading training/val crosswikis dictionary ... ")
        # self.test_kwnen_cwikis = vocabloader.getTestKnwEnCwiki()
        # self.test_allen_cwikis = vocabloader.getTestAllEnCwiki()

        # Crosswikis
        print("Loading Crosswikis dict. (takes ~2 mins to load)")
        self.crosswikis = utils.load(config.crosswikis_pruned_pkl)
        # self.crosswikis = {}
        print("Crosswikis loaded. Size: {}".format(len(self.crosswikis)))

        if self.pretrain_wordembed:
            stime = time.time()
            self.word2vec = vocabloader.loadGloveVectors()
            print("[#] Glove Vectors loaded!")
            ttime = (time.time() - stime)/float(60)
            print("[#] Time to load vectors : {} mins".format(ttime))

        print("[#] Test Mentions File : {}".format(test_mens_file))

        print("[#] Pre-loading test mentions ... ")
        self.mentions = utils.make_mentions_from_file(test_mens_file)
        self.men_idx = 0
        self.num_mens = len(self.mentions)
        self.epochs = 0
        print( "[#] Test Mentions : {}".format(self.num_mens))

        self.batch_size = batch_size
        print("[#] Batch Size: %d" % self.batch_size)
        self.num_cands = num_cands
        self.strict_context = strict_context

        print("\n[#]LOADING COMPLETE")
  # *******************      END __init__      *******************************

    def get_vector(self, word):
        if word in self.word2vec:
            return self.word2vec[word]
        else:
            return self.word2vec['unk']

    def reset_test(self):
        self.men_idx = 0
        self.epochs = 0

    def _read_mention(self):
        mention = self.mentions[self.men_idx]
        self.men_idx += 1
        if self.men_idx == self.num_mens:
            self.men_idx = 0
            self.epochs += 1
        return mention

    def _next_batch(self):
        ''' Data : wikititle \t mid \t wid \t start \t end \t tokens \t labels
        start and end are inclusive
        '''
        # Sentence     = s1 ... m1 ... mN, ... sN.
        # Left Batch   = s1 ... m1 ... mN
        # Right Batch  = sN ... mN ... m1
        (left_batch, right_batch) = ([], [])

        # Labels : Vector of 0s and 1s of size = number of labels = 113
        labels_batch = np.zeros([self.batch_size, self.num_labels])

        coh_indices = []
        coh_values = []
        if self.coherence:
            coh_matshape = [self.batch_size, self.num_cohstr]
        else:
            coh_matshape = []

        # Wiki Description: [B, N=100, D=300]
        # truewid_descvec_batch = []

        # Candidate WID idxs and their cprobs
        # First element is always true wid
        (wid_idxs_batch, wid_cprobs_batch) = ([], [])

        while len(left_batch) < self.batch_size:
            batch_el = len(left_batch)
            m = self._read_mention()

            for label in m.types:
                if label in self.label2idx:
                    labelidx = self.label2idx[label]
                    labels_batch[batch_el][labelidx] = 1.0
            #labels

            cohFound = False    # If no coherence mention is found, then add unk
            if self.coherence:
                cohidxs = []  # Indexes in the [B, NumCoh] matrix
                cohvals = []  # 1.0 to indicate presence
                for cohstr in m.coherence:
                    if cohstr in self.cohG92idx:
                        cohidx = self.cohG92idx[cohstr]
                        cohidxs.append([batch_el, cohidx])
                        cohvals.append(1.0)
                        cohFound = True
                if cohFound:
                    coh_indices.extend(cohidxs)
                    coh_values.extend(cohvals)
                else:
                    cohidx = self.cohG92idx[self.unk_word]
                    coh_indices.append([batch_el, cohidx])
                    coh_values.append(1.0)

            # cohFound = False  # If no coherence mention found, then add unk
            # if self.coherence:
            #     for cohstr in m.coherence:
            #         if cohstr in self.cohG92idx:
            #             cohidx = self.cohG92idx[cohstr]
            #             coh_indices.append([batch_el, cohidx])
            #             coh_values.append(1.0)
            #             cohFound = True
            #             if not cohFound:
            #                 cohidx = self.cohG92idx[self.unk_word]
            #                 coh_indices.append([batch_el, cohidx])
            #                 coh_values.append(1.0)

            # Left and Right context includes mention surface
            left_tokens = m.sent_tokens[0:m.end_token+1]
            right_tokens = m.sent_tokens[m.start_token:][::-1]

            # Strict left and right context
            if self.strict_context:
                left_tokens = m.sent_tokens[0:m.start_token]
                right_tokens = m.sent_tokens[m.end_token+1:][::-1]
            # Left and Right context includes mention surface
            else:
                left_tokens = m.sent_tokens[0:m.end_token+1]
                right_tokens = m.sent_tokens[m.start_token:][::-1]

            if not self.pretrain_wordembed:
                left_idxs = [self.convert_word2idx(word)
                             for word in left_tokens]
                right_idxs = [self.convert_word2idx(word)
                              for word in right_tokens]
            else:
                left_idxs = left_tokens
                right_idxs = right_tokens

            left_batch.append(left_idxs)
            right_batch.append(right_idxs)

            # if m.wid in self.knwid2idx:
            #     truewid_descvec_batch.append(self.kwnwid2descvecs[m.wid])
            # else:
            #     truewid_descvec_batch.append(
            #         self.kwnwid2descvecs[self.unk_wid])

            # wids : [true_knwn_idx, cand1_idx, cand2_idx, ..., unk_idx]
            # wid_cprobs : [cwikis probs or 0.0 for unks]
            (wid_idxs, wid_cprobs) = self.make_candidates_cprobs(m)
            wid_idxs_batch.append(wid_idxs)
            wid_cprobs_batch.append(wid_cprobs)

            # self.print_test_batch(m, wid_idxs, wid_cprobs)
            # print(m.docid)

        #end batch making
        coherence_batch = (coh_indices, coh_values, coh_matshape)

        # return (left_batch, right_batch, truewid_descvec_batch, labels_batch,
        #         coherence_batch, wid_idxs_batch, wid_cprobs_batch)
        return (left_batch, right_batch, labels_batch,
                coherence_batch, wid_idxs_batch, wid_cprobs_batch)

    def print_test_batch(self, mention, wid_idxs, wid_cprobs):
        print("Surface : {}  WID : {}  WT: {}".format(
            mention.surface, mention.wid, self.wid2WikiTitle[mention.wid]))
        print(mention.wid in self.knwid2idx)
        for (idx,cprob) in zip(wid_idxs, wid_cprobs):
            print("({} : {:0.5f})".format(
                self.wid2WikiTitle[self.idx2knwid[idx]], cprob), end=" ")
            print("\n")

    def make_candidates_cprobs(self, m):
        # First wid_idx is true entity
        #if self.useKnownEntitesOnly:
        if m.wid in self.knwid2idx:
            wid_idxs = [self.knwid2idx[m.wid]]
        else:
            wid_idxs = [self.knwid2idx[self.unk_wid]]
        # else:
        #     ''' Todo: Set wids_idxs[0] in a way to incorporate all entities'''
        #     wids_idxs = [0]

        # This prob will be updated when going over cwikis candidates
        wid_cprobs = [0.0]

        # Crosswikis to use based on Known / All entities
        # if self.useKnownEntitesOnly:
        cwiki_dict = self.crosswikis
        # else:
        #    cwiki_dict = self.test_all_cwikis

        # Indexing dict to use
        # Todo: When changed to all entities, indexing will change
        wid2idx = self.knwid2idx

        # Fill num_cands now
        surface = utils._getLnrm(m.surface)
        if surface in cwiki_dict:
            candwids_cprobs = cwiki_dict[surface][0:self.num_cands-1]
            (candwids, candwid_cprobs) = candwids_cprobs
            for (c, p) in zip(candwids, candwid_cprobs):
                if c in wid2idx:
                    if c == m.wid:  # Update cprob for true if in known set
                        wid_cprobs[0] = p
                    else:
                        wid_idxs.append(wid2idx[c])
                        wid_cprobs.append(p)
        # All possible candidates added now. Pad with unks
        assert len(wid_idxs) == len(wid_cprobs)
        remain = self.num_cands - len(wid_idxs)
        wid_idxs.extend([0]*remain)
        wid_cprobs.extend([0.0]*remain)

        wid_idxs = wid_idxs[0:self.num_cands]
        wid_cprobs = wid_cprobs[0:self.num_cands]

        return (wid_idxs, wid_cprobs)

    def embed_batch(self, batch):
        ''' Input is a padded batch of left or right contexts containing words
            Dimensions should be [B, padded_length]
        Output:
            Embed the word idxs using pretrain word embedding
        '''
        output_batch = []
        for sent in batch:
            word_embeddings = [self.get_vector(word) for word in sent]
            output_batch.append(word_embeddings)
        return output_batch

    def embed_mentions_batch(self, mentions_batch):
        ''' Input is batch of mention tokens as a list of list of tokens.
        Output: For each mention, average word embeddings '''
        embedded_mentions_batch = []
        for m_tokens in mentions_batch:
            outvec = np.zeros(300, dtype=float)
            for word in m_tokens:
                outvec += self.get_vector(word)
                outvec = outvec / len(m_tokens)
                embedded_mentions_batch.append(outvec)
        return embedded_mentions_batch

    def pad_batch(self, batch):
        if not self.pretrain_wordembed:
            pad_unit = self.word2idx[self.unk_word]
        else:
            pad_unit = self.unk_word

        lengths = [len(i) for i in batch]
        max_length = max(lengths)
        for i in range(0, len(batch)):
            batch[i].extend([pad_unit]*(max_length - lengths[i]))
        return (batch, lengths)

    def _next_padded_batch(self):
        # (left_batch, right_batch, truewid_descvec_batch,
        #  labels_batch, coherence_batch,
        #  wid_idxs_batch, wid_cprobs_batch) = self._next_batch()
        (left_batch, right_batch,
         labels_batch, coherence_batch,
         wid_idxs_batch, wid_cprobs_batch) = self._next_batch()

        (left_batch, left_lengths) = self.pad_batch(left_batch)
        (right_batch, right_lengths) = self.pad_batch(right_batch)

        if self.pretrain_wordembed:
            left_batch = self.embed_batch(left_batch)
            right_batch = self.embed_batch(right_batch)

        return (left_batch, left_lengths, right_batch, right_lengths,
                labels_batch, coherence_batch,
                wid_idxs_batch, wid_cprobs_batch)

    def convert_word2idx(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx[self.unk_word]

    def next_test_batch(self):
        return self._next_padded_batch()


    def debugWIDIdxsBatch(self, wid_idxs_batch):
        WikiTitles = []
        for widxs in wid_idxs_batch:
            wits = [self.wid2WikiTitle[self.idx2knwid[wididx]] for wididx in widxs]
            WikiTitles.append(wits)

        return WikiTitles

    def widIdx2WikiTitle(self, widIdx):
        wid = self.idx2knwid[widIdx]
        wikiTitle = self.wid2WikiTitle[wid]
        return wikiTitle


if __name__ == '__main__':
    sttime = time.time()
    batch_size = 1
    num_batch = 1000
    configpath = "configs/config.ini"
    config = Config(configpath, verbose=False)
    vocabloader = VocabLoader(config)
    b = TestDataReader(config=config,
                       vocabloader=vocabloader,
                       test_mens_file=config.test_file,
                       num_cands=30,
                       batch_size=batch_size,
                       strict_context=False,
                       pretrain_wordembed=False,
                       coherence=False)

    stime = time.time()

    i = 0
    kwn = 0
    total_instances = 0
    while b.epochs < 1:
        (left_batch, left_lengths,
         right_batch, right_lengths,
         labels_batch, coherence_batch,
         wid_idxs_batch, wid_cprobs_batch) = b.next_test_batch()

        print(b.debugWIDIdxsBatch(wid_idxs_batch))
        print(wid_cprobs_batch)

        if i % 100 == 0:
            etime = time.time()
            t=etime-stime
            print("{} done. Time taken : {} seconds".format(i, t))
            i += 1
    etime = time.time()
    t=etime-stime
    tt = etime - sttime
    print("Batching time (in secs) to make %d batches of size %d : %7.4f seconds" % (i, batch_size, t))
    print("Total time (in secs) to make %d batches of size %d : %7.4f seconds" % (i, batch_size, tt))
