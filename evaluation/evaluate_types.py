import os
import sys
import numpy as np

coarsetypes = set(["location", "person", "organization", "event"])
coarseTypeIds = set([1,5,10,25])

def _convertTypeMatToTypeSets(typesscore_mat, idx2label, threshold):
    ''' Gets true labels and pred scores in numpy matrix and converts to list
    args
      true_label_batch: Binary Numpy matrix of [num_instances, num_labels]
      pred_score_batch: Real [0,1] numpy matrix of [num_instances, num_labels]

    return:
      true_labels: List of list of true label (indices) for batch of instances
      pred_labels : List of list of pred label (indices) for batch of instances
        (threshold = 0.5)
    '''
    labels = []
    for i in typesscore_mat:
        # i in array of label_vals for i-th example
        labels_i = []
        max_idx = -1
        max_val = -1
        for (label_idx, val) in enumerate(i):
            if val >= threshold:
                labels_i.append(idx2label[label_idx])
            if val > max_val:
                max_idx = label_idx
                max_val = val
        if len(labels_i) == 0:
            labels_i.append(idx2label[max_idx])
        labels.append(set(labels_i))

    '''
    assert 0.0 < threshold <= 1.0
    boolmat = typesscore_mat >= threshold
    boollist = boolmat.tolist()
    num_instanes = len(boollist)
    labels = []
    for i in range(0, num_instanes):
      labels_i = [idx2label[i] for i, x in enumerate(boollist[i]) if x]
      labels.append(set(labels_i))
    ##
    '''
    return labels

def convertTypesScoreMatLists_TypeSets(typeScoreMat_list, idx2label, threshold):
    '''
    Take list of type scores numpy mat (per batch) as ouput from Tensorflow.
    Convert into list of type sets for each mention based on the thresold

    Return:
      typeSets_list: Size=num_instances. Each instance is set of type labels for mention
    '''

    typeSets_list = []
    for typeScoreMat in typeScoreMat_list:
        typeLabels_list = _convertTypeMatToTypeSets(typeScoreMat,
                                                    idx2label, threshold)
        typeSets_list.extend(typeLabels_list)
    return typeSets_list


def typesPredictionStats(pred_labels, true_labels):
    '''
    args
      true_label_batch: Binary Numpy matrix of [num_instances, num_labels]
      pred_score_batch: Real [0,1] numpy matrix of [num_instances, num_labels]
    '''

    # t_hat \interesect t
    t_intersect = 0
    t_hat_count = 0
    t_count = 0
    t_t_hat_exact = 0
    loose_macro_p = 0.0
    loose_macro_r = 0.0
    num_instances = len(true_labels)
    for i in range(0, num_instances):
        intersect = len(true_labels[i].intersection(pred_labels[i]))
        t_h_c = len(pred_labels[i])
        t_c = len(true_labels[i])
        t_intersect += intersect
        t_hat_count += t_h_c
        t_count += t_c
        exact = 1 if (true_labels[i] == pred_labels[i]) else 0
        t_t_hat_exact += exact
        if len(pred_labels[i]) > 0:
            loose_macro_p += intersect / float(t_h_c)
        if len(true_labels[i]) > 0:
            loose_macro_r += intersect / float(t_c)

    return (t_intersect, t_t_hat_exact, t_hat_count, t_count,
            loose_macro_p, loose_macro_r)

def typesEvaluationMetrics(pred_TypeSetsList, true_TypeSetsList):
    """
    Return a list of - empirical metrics.

    Args:
        pred_TypeSetsList: (list): write your description
        true_TypeSetsList: (todo): write your description
    """
    num_instances = len(true_TypeSetsList)
    (t_i, t_th_exact, t_h_c, t_c, l_m_p, l_m_r) = typesPredictionStats(
      pred_labels=pred_TypeSetsList, true_labels=true_TypeSetsList)
    strict = float(t_th_exact)/float(num_instances)
    loose_macro_p = l_m_p / float(num_instances)
    loose_macro_r = l_m_r / float(num_instances)
    loose_macro_f = f1(loose_macro_p, loose_macro_r)
    if t_h_c > 0:
        loose_micro_p = float(t_i)/float(t_h_c)
    else:
        loose_micro_p = 0
    if t_c > 0:
        loose_micro_r = float(t_i)/float(t_c)
    else:
        loose_micro_r = 0
    loose_micro_f = f1(loose_micro_p, loose_micro_r)

    return (strict, loose_macro_p, loose_macro_r, loose_macro_f, loose_micro_p,
            loose_micro_r, loose_micro_f)


def performTypingEvaluation(predLabelScoresnumpymat_list, idx2label):
    '''
    Args: List of numpy mat, one for ech batch, for true and pred type scores
      trueLabelScoresnumpymat_list: List of score matrices output by tensorflow
      predLabelScoresnumpymat_list: List of score matrices output by tensorflow
    '''
    pred_TypeSetsList = convertTypesScoreMatLists_TypeSets(
        typeScoreMat_list=predLabelScoresnumpymat_list, idx2label=idx2label,
        threshold=0.75)

    return pred_TypeSetsList


def evaluate(predLabelScoresnumpymat_list, idx2label):
    """
    Evaluate the classification of a list of label.

    Args:
        predLabelScoresnumpymat_list: (int): write your description
        idx2label: (str): write your description
    """
    # print("Evaluating Typing ... ")
    pred_TypeSetsList = convertTypesScoreMatLists_TypeSets(
        typeScoreMat_list=predLabelScoresnumpymat_list, idx2label=idx2label,
        threshold=0.75)

    return pred_TypeSetsList


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
