import os
import sys
import numpy as np


def computeMaxPriorContextJointEntities(
    WIDS_list, wikiTitles_list, condProbs_list, contextProbs_list,
    condContextJointProbs_list, verbose):
    """
    Computes a list of cycles of a list of tuples ).

    Args:
        WIDS_list: (list): write your description
        wikiTitles_list: (list): write your description
        condProbs_list: (list): write your description
        contextProbs_list: (list): write your description
        condContextJointProbs_list: (todo): write your description
        verbose: (int): write your description
    """

    assert (len(wikiTitles_list) == len(condProbs_list) ==
            len(contextProbs_list) == len(condContextJointProbs_list))
    numMens = len(wikiTitles_list)

    evaluationWikiTitles = []
    sortedContextWTs = []
    for (WIDS, wTs,
         cProbs, contProbs, jointProbs) in zip(WIDS_list,
                                               wikiTitles_list,
                                               condProbs_list,
                                               contextProbs_list,
                                               condContextJointProbs_list):
        # if wTs[0] == "<unk_wid>":
        #     evaluationWikiTitles.append([tuple(["<unk_wid>"]*3),
        #                                  tuple(["<unk_wid>"]*3)])
        # else:
        maxCondEntity_idx = np.argmax(cProbs)
        maxCondWID = WIDS[maxCondEntity_idx]
        maxCondEntity = wTs[maxCondEntity_idx]
        maxCondProb = cProbs[maxCondEntity_idx]

        maxContEntity_idx = np.argmax(contProbs)
        maxContWID = WIDS[maxContEntity_idx]
        maxContEntity = wTs[maxContEntity_idx]
        maxContProb = contProbs[maxContEntity_idx]

        contProbs_sortIdxs = np.argsort(contProbs).tolist()[::-1]
        sortContWTs = [wTs[i] for i in contProbs_sortIdxs]
        sortedContextWTs.append(sortContWTs)

        maxJointEntity_idx = np.argmax(jointProbs)
        maxJointWID = WIDS[maxJointEntity_idx]
        maxJointEntity = wTs[maxJointEntity_idx]
        maxJointProb = jointProbs[maxJointEntity_idx]
        maxJointCprob = cProbs[maxJointEntity_idx]
        maxJointContP = contProbs[maxJointEntity_idx]

        predWTs = (maxCondEntity, maxContEntity, maxJointEntity)
        predWIDs = (maxCondWID, maxContWID, maxJointWID)
        predProbs = (maxCondProb, maxContProb, maxJointProb)
        evaluationWikiTitles.append([predWTs, predWIDs, predProbs])

    assert len(evaluationWikiTitles) == numMens

    return (evaluationWikiTitles, sortedContextWTs)

def convertWidIdxs2WikiTitlesAndWIDs(widIdxs_list, idx2knwid, wid2WikiTitle):
    """
    Convert widxs to widxs.

    Args:
        widIdxs_list: (list): write your description
        idx2knwid: (int): write your description
        wid2WikiTitle: (int): write your description
    """
    wikiTitles_list = []
    WIDS_list = []
    for widIdxs in widIdxs_list:
        wids = [idx2knwid[wididx] for wididx in widIdxs]
        wikititles = [wid2WikiTitle[idx2knwid[wididx]] for wididx in widIdxs]
        WIDS_list.append(wids)
        wikiTitles_list.append(wikititles)

    return (WIDS_list, wikiTitles_list)

def _normalizeProbList(probList):
    """
    Normalize probabilities.

    Args:
        probList: (list): write your description
    """
    norm_probList = []
    for probs in probList:
        s = sum(probs)
        if s != 0.0:
            n_p = [p/s for p in probs]
            norm_probList.append(n_p)
        else:
            norm_probList.append(probs)
    return norm_probList


def computeFinalEntityProbs(condProbs_list, contextProbs_list):
    """
    Compute a list of probabilities.

    Args:
        condProbs_list: (list): write your description
        contextProbs_list: (list): write your description
    """
    condContextJointProbs_list = []
    condProbs_list = _normalizeProbList(condProbs_list)
    contextProbs_list = _normalizeProbList(contextProbs_list)

    for (cprobs, contprobs) in zip(condProbs_list, contextProbs_list):
        condcontextprobs = [(x + y - x*y) for (x,y) in zip(cprobs, contprobs)]
        sum_condcontextprobs = sum(condcontextprobs)
        if sum_condcontextprobs != 0.0:
            condcontextprobs = [float(x)/sum_condcontextprobs for x in condcontextprobs]
        condContextJointProbs_list.append(condcontextprobs)
    return condContextJointProbs_list


def computeFinalEntityScores(condProbs_list, contextProbs_list, alpha=0.5):
    """
    Computes the probabilities of a list.

    Args:
        condProbs_list: (list): write your description
        contextProbs_list: (list): write your description
        alpha: (float): write your description
    """
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


#############################################################################


def evaluateEL(condProbs_list, widIdxs_list, contextProbs_list,
               idx2knwid, wid2WikiTitle, verbose=False):
    ''' Prior entity prob, True and candidate entity WIDs, Predicted ent. prob.
    using context for each of te 30 candidates. First element in the candidates
    is the true entity.
    Args:
      For each mention:
        condProbs_list: List of prior probs for 30 candidates.
        widIdxs_list: List of candidate widIdxs probs for 30 candidates.
        contextProbss_list: List of candidate prob. using context
      idx2knwid: Map for widIdx -> WID
      wid2WikiTitle: Map from WID -> WikiTitle
    '''
    # print("Evaluating E-Linking ... ")
    (WIDS_list, wikiTitles_list) = convertWidIdxs2WikiTitlesAndWIDs(
      widIdxs_list, idx2knwid, wid2WikiTitle)

    jointProbs_list = computeFinalEntityProbs(condProbs_list,
                                              contextProbs_list)

    (evaluationWikiTitles,
     sortedContextWTs) = computeMaxPriorContextJointEntities(
        WIDS_list, wikiTitles_list, condProbs_list, contextProbs_list,
        jointProbs_list, verbose)

    return (jointProbs_list, evaluationWikiTitles, sortedContextWTs)

##############################################################################


def f1(p,r):
    """
    Convert f1 ( r2

    Args:
        p: (int): write your description
        r: (int): write your description
    """
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
    """
    Corrects the predicted_posterior.

    Args:
        entity_posterior_scores: (todo): write your description
        batch_size: (int): write your description
    """
    bool_array = np.equal(np.argmax(entity_posterior_scores, axis=1),
                          [0]*batch_size)
    correct_preds = np.sum(bool_array)
    return correct_preds
