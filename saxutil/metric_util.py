import numpy as np


def mk_lbl_metric_dict(tgts, preds, metrics):
    '''
    :type tgts: list<obj>
    :param tgts: list of hashable objects, generally numbers or strings
    :type preds: list<obj>
    :param preds: list of hashable objects, generally numbers or strings
    :type metric: function
    :param metric: generally a precision, recall, f1, or accuracy metric
        function.
    '''
    assert(len(tgts) == len(preds))
    sorted_lbls = sorted(list(set(tgts).union(preds)))

    metric_arrs = []
    for metric in metrics:
        arr = metric(tgts, preds, average=None)
        metric_arrs.append(arr)

    assert(len(arr) == len(sorted_lbls))
    md = {}
    for i in range(len(sorted_lbls)):
        scores = []
        for j in range(len(metric_arrs)):
            scores.append(metric_arrs[j][i])
        md[sorted_lbls[i]] = scores
    return md


def find_problem_cats_from_metric_dict(metric_dict, power=.5):
    tups = []
    for i in metric_dict.items():
        problem_score = (1.0 - i[1][0]) * np.power(i[1][1], power)
        tups.append(
            (i[0], i[1][0], i[1][1], problem_score,)
        )
    tups.sort(key=lambda t: t[-1])
    tups.reverse()
    return tups
