import sys
import time
import numpy as np
import readers.utils as utils
from readers.Mention import Mention
from readers.config import Config
from readers.vocabloader import VocabLoader
import ccg_nlpy
from ccg_nlpy.core.text_annotation import TextAnnotation

start_word = "<s>"
end_word = "<eos>"

# Reader for Text Annotations
class TextAnnoTestReader(object):
    def __init__(self, config, vocabloader,
                 num_cands, batch_size, strict_context=True,
                 pretrain_wordembed=True, coherence=True,
                 nerviewname="NER_CONLL"):
        """
        Initialize word embeddings.

        Args:
            self: (todo): write your description
            config: (todo): write your description
            vocabloader: (todo): write your description
            num_cands: (int): write your description
            batch_size: (int): write your description
            strict_context: (str): write your description
            pretrain_wordembed: (bool): write your description
            coherence: (todo): write your description
            nerviewname: (str): write your description
        """
        self.typeOfReader = "inference"
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = 'unk'  # In tune with word2vec
        self.unk_wid = "<unk_wid>"
        self.tr_sup = 'tr_sup'
        self.tr_unsup = 'tr_unsup'
        self.pretrain_wordembed = pretrain_wordembed
        self.coherence = coherence
        self.nerviewname = nerviewname

        # Word Vocab
        (self.word2idx, self.idx2word) = vocabloader.getGloveWordVocab()
        self.num_words = len(self.idx2word)

        # Label Vocab
        (self.label2idx, self.idx2label) = vocabloader.getLabelVocab()
        self.num_labels = len(self.idx2label)

        # Known WID Vocab
        (self.knwid2idx, self.idx2knwid) = vocabloader.getKnwnWidVocab()
        self.num_knwn_entities = len(self.idx2knwid)

        # Wid2Wikititle Map
        self.wid2WikiTitle = vocabloader.getWID2Wikititle()

        # Coherence String Vocab
        print("Loading Coherence Strings Dicts ... ")
        (self.cohG92idx, self.idx2cohG9) = utils.load(
            config.cohstringG9_vocab_pkl)
        self.num_cohstr = len(self.idx2cohG9)

        # Crosswikis
        print("Loading Crosswikis dict. (takes ~2 mins to load)")
        self.crosswikis = utils.load(config.crosswikis_pruned_pkl)
        print("Crosswikis loaded. Size: {}".format(len(self.crosswikis)))

        if self.pretrain_wordembed:
            stime = time.time()
            self.word2vec = vocabloader.loadGloveVectors()
            print("[#] Glove Vectors loaded!")
            ttime = (time.time() - stime)/float(60)


        # print("[#] Test Mentions File : {}".format(test_mens_file))

        # print("[#] Loading test file and preprocessing ... ")
        # with open(test_mens_file, 'r') as f:
        #     tajsonstr = f.read()
        # ta = TextAnnotation(json_str=tajsonstr)
        #
        # (sentences_tokenized, modified_ner_cons_list) = self.processTestDoc(ta)
        #
        # self.mention_lines = self.convertSent2NerToMentionLines(
        #     sentences_tokenized, modified_ner_cons_list)
        #
        # self.mentions = []
        # for line in self.mention_lines:
        #     m = Mention(line)
        #     self.mentions.append(m)

        self.men_idx = 0
        # self.num_mens = len(self.mentions)
        self.epochs = 0
        # print( "[#] Test Mentions : {}".format(self.num_mens))

        self.batch_size = batch_size
        print("[#] Batch Size: %d" % self.batch_size)
        self.num_cands = num_cands
        self.strict_context = strict_context

        print("\n[#]LOADING COMPLETE")
  #*******************      END __init__      *********************************

    def new_test_file(self, test_mens_file):
        """
        Create a new test file.

        Args:
            self: (todo): write your description
            test_mens_file: (str): write your description
        """
        self.test_mens_file = test_mens_file

        with open(test_mens_file, 'r') as f:
            tajsonstr = f.read()
        ta = TextAnnotation(json_str=tajsonstr)
        self.textanno = ta

        (sentences_tokenized, modified_ner_cons_list) = self.processTestDoc(ta)

        self.mention_lines = self.convertSent2NerToMentionLines(
            sentences_tokenized, modified_ner_cons_list)

        self.mentions = []
        for line in self.mention_lines:
            m = Mention(line)
            self.mentions.append(m)

        self.men_idx = 0
        self.num_mens = len(self.mentions)
        self.epochs = 0

    def new_tajsonstr(self, tajsonstr):
        """ tajsonstr is a json str of a TA """
        ta = TextAnnotation(json_str=tajsonstr)
        self.new_ta(ta)

    def new_ta(self, ta):
        """
        Create a new mention.

        Args:
            self: (todo): write your description
            ta: (todo): write your description
        """
        self.textanno = ta

        (sentences_tokenized, modified_ner_cons_list) = self.processTestDoc(ta)

        self.mention_lines = self.convertSent2NerToMentionLines(
            sentences_tokenized, modified_ner_cons_list)

        self.mentions = []
        for line in self.mention_lines:
            m = Mention(line)
            self.mentions.append(m)

        self.men_idx = 0
        self.num_mens = len(self.mentions)
        self.epochs = 0



    def get_vector(self, word):
        """
        Get vector of word.

        Args:
            self: (todo): write your description
            word: (str): write your description
        """
        if word in self.word2vec:
            return self.word2vec[word]
        else:
            return self.word2vec['unk']

    def reset_test(self):
        """
        !

        Args:
            self: (todo): write your description
        """
        self.men_idx = 0
        self.epochs = 0

    def processTestDoc(self, ccgdoc):
        """
        Given a list of sentences : list of sentences

        Args:
            self: (todo): write your description
            ccgdoc: (todo): write your description
        """
        doc_tokens = ccgdoc.get_tokens
        # sent_end_token_indices : contains index for the starting of the
        # next sentence.
        sent_end_token_indices = \
            ccgdoc.get_sentence_end_token_indices
        # List of tokenized sentences
        sentences_tokenized = []
        for i in range(0, len(sent_end_token_indices)):
            start = sent_end_token_indices[i-1] if i != 0 else 0
            end = sent_end_token_indices[i]
            sent_tokens = doc_tokens[start:end]
            sentences_tokenized.append(sent_tokens)

        # List of ner dicts from ccg pipeline
        ner_cons_list = []
        try:
            ner_cons_list = ccgdoc.get_view(self.nerviewname).cons_list
        except:
            print("NO NAMED ENTITIES IN THE DOC. EXITING")

        modified_ner_cons_list = []

        for orig_ner in ner_cons_list:
            ner = orig_ner.copy()
            # ner['end'] = ner['end'] + 1
            # ner['tokens'] = ' '.join(doc_tokens[ner['start']:ner['end']])

            found = False
            # idx = sentIdx, j = sentEndTokenIdx
            for idx, j in enumerate(sent_end_token_indices):
                sent_start_token = sent_end_token_indices[idx-1] \
                    if idx != 0 else 0
                # ner['end'] is the idx of the token after ner
                if ner['end'] <= j:
                    ner['start'] = ner['start'] - sent_start_token
                    ner['end'] = ner['end'] - sent_start_token - 1
                    ner['sent_idx'] = idx

                    modified_ner_cons_list.append(ner)

                    found = True
                if found:
                    break
        return (sentences_tokenized, modified_ner_cons_list)

    def convertSent2NerToMentionLines(self, sentences_tokenized,
                                      modified_ner_cons_list):
        '''Convert NERs from document to list of mention strings'''
        mentions = []
        # Make Document Context String for whole document
        cohStr = ""
        # for sent_idx, s_nerDicts in sentidx2ners.items():
        #     for s, ner in s_nerDicts:
        #         cohStr += ner['tokens'].replace(' ', '_') + ' '

        for ner_men in modified_ner_cons_list:
            cohStr += ner_men['tokens'].replace(' ', '_') + ' '

        cohStr = cohStr.strip()

        for ner_men in modified_ner_cons_list:
            idx = ner_men['sent_idx']
            sentence = ' '.join(sentences_tokenized[idx])

            mention = "%s\t%s\t%s" % ("unk_mid", "unk_wid", "unkWT")
            mention = mention + '\t' + str(ner_men['start'])
            mention = mention + '\t' + str(ner_men['end'])
            mention = mention + '\t' + str(ner_men['tokens'])
            mention = mention + '\t' + sentence
            mention = mention + '\t' + "UNK_TYPES"
            mention = mention + '\t' + cohStr
            mentions.append(mention)
        return mentions

    def bracketMentionInSentence(self, s, nerDict):
        """
        Generate a string for a - delimiterable.

        Args:
            self: (todo): write your description
            s: (str): write your description
            nerDict: (todo): write your description
        """
        tokens = s.split(" ")
        start = nerDict['start']
        end = nerDict['end']
        tokens.insert(start, '[[')
        tokens.insert(end + 2, ']]')
        return ' '.join(tokens)

    def _read_mention(self):
        """
        Reads the next epoch.

        Args:
            self: (todo): write your description
        """
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

        coh_indices = []
        coh_values = []
        if self.coherence:
            coh_matshape = [self.batch_size, self.num_cohstr]
        else:
            coh_matshape = []

        # Candidate WID idxs and their cprobs
        # First element is always true wid
        (wid_idxs_batch, wid_cprobs_batch) = ([], [])

        while len(left_batch) < self.batch_size:
            batch_el = len(left_batch)
            m = self._read_mention()

            # for label in m.types:
            #     if label in self.label2idx:
            #         labelidx = self.label2idx[label]
            #         labels_batch[batch_el][labelidx] = 1.0

            cohFound = False    # If no coherence mention is found, add unk
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

            # wids : [true_knwn_idx, cand1_idx, cand2_idx, ..., unk_idx]
            # wid_cprobs : [cwikis probs or 0.0 for unks]
            (wid_idxs, wid_cprobs) = self.make_candidates_cprobs(m)
            wid_idxs_batch.append(wid_idxs)
            wid_cprobs_batch.append(wid_cprobs)

        coherence_batch = (coh_indices, coh_values, coh_matshape)

        return (left_batch, right_batch,
                coherence_batch, wid_idxs_batch, wid_cprobs_batch)

    def print_test_batch(self, mention, wid_idxs, wid_cprobs):
        """
        Print a batch of the test results.

        Args:
            self: (todo): write your description
            mention: (str): write your description
            wid_idxs: (str): write your description
            wid_cprobs: (str): write your description
        """
        print("Surface : {}  WID : {}  WT: {}".format(
            mention.surface, mention.wid, self.wid2WikiTitle[mention.wid]))
        print(mention.wid in self.knwid2idx)
        for (idx,cprob) in zip(wid_idxs, wid_cprobs):
            print("({} : {:0.5f})".format(
                self.wid2WikiTitle[self.idx2knwid[idx]], cprob), end=" ")
            print("\n")

    def make_candidates_cprobs(self, m):
        """
        Make candidate candidates.

        Args:
            self: (todo): write your description
            m: (todo): write your description
        """
        # Fill num_cands now
        surface = utils._getLnrm(m.surface)
        wid_idxs = []
        wid_cprobs = []

        # print(surface)
        if surface in self.crosswikis:
            # Pruned crosswikis has only known wids and 30 cands at max
            candwids_cprobs = self.crosswikis[surface][0:self.num_cands-1]
            (wids, wid_cprobs) = candwids_cprobs
            wid_idxs = [self.knwid2idx[wid] for wid in wids]

        # All possible candidates added now. Pad with unks

        # assert len(wid_idxs) == len(wid_cprobs)
        remain = self.num_cands - len(wid_idxs)
        wid_idxs.extend([0]*remain)
        remain = self.num_cands - len(wid_cprobs)
        wid_cprobs.extend([0.0]*remain)

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
        """
        Pad a batch of words into a batch of words.

        Args:
            self: (todo): write your description
            batch: (todo): write your description
        """
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
        """
        Batch next batch.

        Args:
            self: (todo): write your description
        """
        (left_batch, right_batch,
         coherence_batch,
         wid_idxs_batch, wid_cprobs_batch) = self._next_batch()

        (left_batch, left_lengths) = self.pad_batch(left_batch)
        (right_batch, right_lengths) = self.pad_batch(right_batch)

        if self.pretrain_wordembed:
            left_batch = self.embed_batch(left_batch)
            right_batch = self.embed_batch(right_batch)

        return (left_batch, left_lengths, right_batch, right_lengths,
                coherence_batch, wid_idxs_batch, wid_cprobs_batch)

    def convert_word2idx(self, word):
        """
        Converts word2idx.

        Args:
            self: (todo): write your description
            word: (str): write your description
        """
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx[self.unk_word]

    def next_test_batch(self):
        """
        Returns the next batch.

        Args:
            self: (todo): write your description
        """
        return self._next_padded_batch()

    def widIdx2WikiTitle(self, widIdx):
        """
        Return the w : py : class : wx2Wiki.

        Args:
            self: (todo): write your description
            widIdx: (int): write your description
        """
        wid = self.idx2knwid[widIdx]
        wikiTitle = self.wid2WikiTitle[wid]
        return wikiTitle

if __name__ == '__main__':
    sttime = time.time()
    batch_size = 2
    num_batch = 1000
    configpath = "configs/all_mentions_config.ini"
    config = Config(configpath, verbose=False)
    vocabloader = VocabLoader(config)
    b = TextAnnoTestReader(config=config,
                           vocabloader=vocabloader,
                           num_cands=30,
                           batch_size=batch_size,
                           strict_context=False,
                           pretrain_wordembed=True,
                           coherence=True)

    stime = time.time()

    i = 0
    total_instances = 0
    while b.epochs < 1:
        (left_batch, left_lengths, right_batch, right_lengths,
         coherence_batch, wid_idxs_batch,
         wid_cprobs_batch) = b.next_test_batch()
        if i % 100 == 0:
            etime = time.time()
            t=etime-stime
            print("{} done. Time taken : {} seconds".format(i, t))
            i += 1
    etime = time.time()
    t=etime-stime
    tt = etime - sttime
    print("Total Instances : {}".format(total_instances))
    print("Batching time (in secs) to make %d batches of size %d : %7.4f seconds" % (i, batch_size, t))
    print("Total time (in secs) to make %d batches of size %d : %7.4f seconds" % (i, batch_size, tt))
