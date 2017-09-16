import os
import sys
import time
from readers.config import Config
from readers.vocabloader import VocabLoader
from readers import utils



class CrosswikisTest(object):
    def __init__(self, config, vocabloader):
        print("Loading Crosswikis")
        # self.crosswikis = vocabloader.loadCrosswikis()

        stime = time.time()
        self.crosswikis = utils.load(config.crosswikis_pruned_pkl)
        ttime = time.time() - stime
        print("Crosswikis Loaded. Size : {}".format(len(self.crosswikis)))
        print("Time taken : {} secs".format(ttime))

        (self.knwid2idx, self.idx2knwid) = vocabloader.getKnwnWidVocab()
        print("Size of known wids : {}".format(len(self.knwid2idx)))

    def test(self):
        print("Test starting")
        maxCands = 0
        minCands = 0
        notKnownWid = False
        smallSurface = 0
        notSortedProbs = 0
        for surface, c_cprobs in self.crosswikis.items():
            notSorted = False
            numCands = len(c_cprobs)
            if numCands < minCands:
                minCands = numCands
            if numCands > maxCands:
                maxCands = numCands

            prob_prv = 10.0
            for (wid, prob) in c_cprobs:
                if wid not in self.knwid2idx:
                    notKnownWid = True
                if prob_prv < prob:
                    notSorted = True
                prob_prv = prob

            if notSorted:
                notSortedProbs += 1

            if len(surface) <= 1:
                smallSurface += 1

        print("Max Cands : {}".format(maxCands))
        print("min Cands : {}".format(minCands))
        print("Not Known Wid : {}".format(notKnownWid))
        print("small surfaces {}".format(smallSurface))
        print("Not Sorted Probs {}".format(notSortedProbs))

    def test_pruned(self):
        print("Test starting")
        maxCands = 0
        minCands = 30
        notKnownWid = False
        smallSurface = 0
        notSortedProbs = 0
        for surface, c_cprobs in self.crosswikis.items():
            notSorted = False
            (wids, probs) = c_cprobs
            numCands = len(wids)
            if numCands < minCands:
                minCands = numCands
            if numCands > maxCands:
                maxCands = numCands

            prob_prv = 10.0
            for (wid, prob) in zip(wids, probs):
                if wid not in self.knwid2idx:
                    notKnownWid = True
                if prob_prv < prob:
                    notSorted = True
                prob_prv = prob

            if notSorted:
                notSortedProbs += 1

            if len(surface) <= 1:
                smallSurface += 1

        print("Max Cands : {}".format(maxCands))
        print("min Cands : {}".format(minCands))
        print("Not Known Wid : {}".format(notKnownWid))
        print("small surfaces {}".format(smallSurface))
        print("Not Sorted Probs {}".format(notSortedProbs))

    def makeCWKnown(self, cwOutPath):
        cw = {}
        MAXCAND = 30
        surfacesProcessed = 0
        for surface, c_cprobs in self.crosswikis.items():
            surfacesProcessed += 1
            if surfacesProcessed % 1000000 == 0:
                print("Surfaces Processed : {}".format(surfacesProcessed))

            if len(c_cprobs) == 0:
                continue
            if len(surface) <= 1:
                continue
            candsAdded = 0
            c_probs = ([], [])
            # cw[surface] = ([], [])
            for (wid, prob) in c_cprobs:
                if candsAdded == 30:
                    break
                if wid in self.knwid2idx:
                    c_probs[0].append(wid)
                    c_probs[1].append(prob)
                    candsAdded += 1
            if candsAdded != 0:
                cw[surface] = c_probs
        print("Processed")
        print("Size of CW : {}".format(len(cw)))
        utils.save(cwOutPath, cw)
        print("Saved pruned CW")



if __name__ == '__main__':
    configpath = "configs/config.ini"
    config = Config(configpath, verbose=False)
    vocabloader = VocabLoader(config)
    cwikistest = CrosswikisTest(config, vocabloader)
    cwikistest.test_pruned()
    # cwikistest.makeCWKnown(os.path.join(config.resources_dir,
    #                                     "crosswikis.pruned.pkl"))
