import os
import sys
import numpy as np

coarsetypes = set(["location", "person", "organization", "event"])
coarseTypeIds = set([1,5,10,25])


def computeMaxPriorContextJointEntities(
    WIDS_list, wikiTitles_list, condProbs_list, contextProbs_list,
    condContextJointProbs_list, verbose):

    assert (len(wikiTitles_list) == len(condProbs_list) ==
            len(contextProbs_list) == len(condContextJointProbs_list))
    numMens = len(wikiTitles_list)
    numWithCorrectInCand = 0
    accCond = 0
    accCont = 0
    accJoint = 0

  # [[(trueWT, maxPrWT, maxContWT, maxJWT), (trueWID, maxPrWID, maxContWID, maxJWID)]]
    evaluationWikiTitles = []

    sortedContextWTs = []

    for (WIDS, wTs, cProbs, contProbs, jointProbs) in zip(WIDS_list,
                                                          wikiTitles_list,
                                                          condProbs_list,
                                                          contextProbs_list,
                                                          condContextJointProbs_list):
        if wTs[0] == "<unk_wid>":
            evaluationWikiTitles.append([tuple(["<unk_wid>"]*4), tuple(["<unk_wid>"]*4)])

        else:
            numWithCorrectInCand += 1
            trueWID = WIDS[0]
            trueEntity = wTs[0]
            tCondProb = cProbs[0]
            tContProb = contProbs[0]
            tJointProb = jointProbs[0]

            maxCondEntity_idx = np.argmax(cProbs)
            maxCondWID = WIDS[maxCondEntity_idx]
            maxCondEntity = wTs[maxCondEntity_idx]
            maxCondProb = cProbs[maxCondEntity_idx]
            if trueEntity == maxCondEntity and maxCondProb!=0.0:
                accCond+= 1

            maxContEntity_idx = np.argmax(contProbs)
            maxContWID = WIDS[maxContEntity_idx]
            maxContEntity = wTs[maxContEntity_idx]
            maxContProb = contProbs[maxContEntity_idx]
            if maxContEntity == trueEntity and maxContProb!=0.0:
                accCont+= 1

            contProbs_sortIdxs = np.argsort(contProbs).tolist()[::-1]
            sortContWTs = [wTs[i] for i in contProbs_sortIdxs]
            sortedContextWTs.append(sortContWTs)

            maxJointEntity_idx = np.argmax(jointProbs)
            maxJointWID = WIDS[maxJointEntity_idx]
            maxJointEntity = wTs[maxJointEntity_idx]
            maxJointProb = jointProbs[maxJointEntity_idx]
            maxJointCprob = cProbs[maxJointEntity_idx]
            maxJointContP = contProbs[maxJointEntity_idx]
            if maxJointEntity == trueEntity and maxJointProb!=0:
                accJoint+= 1

            predWTs = (trueEntity, maxCondEntity, maxContEntity, maxJointEntity)
            predWIDs = (trueWID, maxCondWID, maxContWID, maxJointWID)
            evaluationWikiTitles.append([predWTs, predWIDs])

        if verbose:
            print("True: {}  c:{:.3f} cont:{:.3f} J:{:.3f}".format(
              trueEntity, tCondProb, tContProb, tJointProb))
            print("Pred: {}  c:{:.3f} cont:{:.3f} J:{:.3f}".format(
              maxJointEntity, maxJointCprob, maxJointContP, maxJointProb))
            print("maxPrior: {}  p:{:.3f} maxCont:{} p:{:.3f}".format(
              maxCondEntity, maxCondProb, maxContEntity, maxContProb))
    #AllMentionsProcessed
    if numWithCorrectInCand != 0:
        accCond = accCond/float(numWithCorrectInCand)
        accCont = accCont/float(numWithCorrectInCand)
        accJoint = accJoint/float(numWithCorrectInCand)
    else:
        accCond = 0.0
        accCont = 0.0
        accJoint = 0.0

    print("Total Mentions : {} In Knwn Mentions : {}".format(
      numMens, numWithCorrectInCand))
    print("Priors Accuracy: {:.3f}  Context Accuracy: {:.3f}  Joint Accuracy: {:.3f}".format(
        (accCond), accCont, accJoint))

    assert len(evaluationWikiTitles) == numMens

    return (evaluationWikiTitles, sortedContextWTs)


def convertWidIdxs2WikiTitlesAndWIDs(widIdxs_list, idx2knwid, wid2WikiTitle):
    wikiTitles_list = []
    WIDS_list = []
    for widIdxs in widIdxs_list:
        wids = [idx2knwid[wididx] for wididx in widIdxs]
        wikititles = [wid2WikiTitle[idx2knwid[wididx]] for wididx in widIdxs]
        WIDS_list.append(wids)
        wikiTitles_list.append(wikititles)

    return (WIDS_list, wikiTitles_list)


def _normalizeProbList(probList):
    norm_probList = []
    for probs in probList:
        s = sum(probs)
        if s != 0.0:
            n_p = [p/s for p in probs]
            norm_probList.append(n_p)
        else:
            norm_probList.append(probs)
    return norm_probList


def computeFinalEntityProbs(condProbs_list, contextProbs_list, alpha=0.5):
    condContextJointProbs_list = []
    condProbs_list = _normalizeProbList(condProbs_list)
    contextProbs_list = _normalizeProbList(contextProbs_list)

    for (cprobs, contprobs) in zip(condProbs_list, contextProbs_list):
        #condcontextprobs = [(alpha*x + (1-alpha)*y) for (x,y) in zip(cprobs, contprobs)]
        condcontextprobs = [(x + y - x*y) for (x,y) in zip(cprobs, contprobs)]
        sum_condcontextprobs = sum(condcontextprobs)
        if sum_condcontextprobs != 0.0:
            condcontextprobs = [float(x)/sum_condcontextprobs for x in condcontextprobs]
        condContextJointProbs_list.append(condcontextprobs)
    return condContextJointProbs_list


def computeFinalEntityScores(condProbs_list, contextProbs_list, alpha=0.5):
    condContextJointProbs_list = []
    condProbs_list = _normalizeProbList(condProbs_list)
    #contextProbs_list = _normalizeProbList(contextProbs_list)

    for (cprobs, contprobs) in zip(condProbs_list, contextProbs_list):
        condcontextprobs = [(alpha*x + (1-alpha)*y) for (x,y) in zip(cprobs, contprobs)]
        sum_condcontextprobs = sum(condcontextprobs)
        if sum_condcontextprobs != 0.0:
            condcontextprobs = [float(x)/sum_condcontextprobs for x in condcontextprobs]
        condContextJointProbs_list.append(condcontextprobs)
    return condContextJointProbs_list


##############################################################################

def evaluateEL(condProbs_list, widIdxs_list, contextProbs_list,
               idx2knwid, wid2WikiTitle, verbose=False):
    ''' Prior entity prob, True and candidate entity WIDs, Predicted ent. prob.
    using context for each of te 30 candidates. First element in the candidates is
    the true entity.
    Args:
      For each mention:
        condProbs_list: List of prior probs for 30 candidates.
        widIdxs_list: List of candidate widIdxs probs for 30 candidates.
        contextProbss_list: List of candidate prob. using context
      idx2knwid: Map for widIdx -> WID
      wid2WikiTitle: Map from WID -> WikiTitle
      wid2TypeLabels: Map from WID -> List of Types
    '''
    print("Evaluating E-Linking ... ")
    (WIDS_list, wikiTitles_list) = convertWidIdxs2WikiTitlesAndWIDs(
      widIdxs_list, idx2knwid, wid2WikiTitle)

    alpha = 0.5
    #for alpha in alpha_range:
    print("Alpha : {}".format(alpha))
    jointProbs_list = computeFinalEntityProbs(
      condProbs_list, contextProbs_list, alpha=alpha)

    # evaluationWikiTitles:
    #     For each mention [(trWT, maxPWT, maxCWT, maxJWT), (trWID, ...)]
    (evaluationWikiTitles,
     sortedContextWTs) = computeMaxPriorContextJointEntities(
        WIDS_list, wikiTitles_list, condProbs_list, contextProbs_list,
        jointProbs_list, verbose)


    '''
    condContextJointScores_list = computeFinalEntityScores(
      condProbs_list, contextProbs_list, alpha=alpha)

    evaluationWikiTitles = computeMaxPriorContextJointEntities(
      WIDS_list, wikiTitles_list, condProbs_list, contextProbs_list,
      condContextJointScores_list, verbose)
    '''

    return (jointProbs_list, evaluationWikiTitles, sortedContextWTs)

##############################################################################


def f1(p,r):
    if p == 0.0 and r == 0.0:
        return 0.0
    return (float(2*p*r))/(p + r)


def strict_pred(true_label_batch, pred_score_batch):
    ''' Calculates strict precision/recall/f1 given truth and predicted scores
    args
      true_label_batch: Binary Numpy matrix of [num_instances, num_labels]
      pred_score_batch: Real [0,1] numpy matrix of [num_instances, num_labels]

    return:
      correct_preds: Number of correct strict preds
      precision : correct_preds / num_instances
    '''
    (true_labels, pred_labels) = types_convert_mat_to_sets(
      true_label_batch, pred_score_batch)

    num_instanes = len(true_labels)
    correct_preds = 0
    for i in range(0, num_instanes):
        if true_labels[i] == pred_labels[i]:
            correct_preds += 1
    #endfor
    precision = recall = float(correct_preds)/num_instanes

    return correct_preds, precision


def correct_context_prediction(entity_posterior_scores, batch_size):
    bool_array = np.equal(np.argmax(entity_posterior_scores, axis=1),
                          [0]*batch_size)
    correct_preds = np.sum(bool_array)
    return correct_preds
