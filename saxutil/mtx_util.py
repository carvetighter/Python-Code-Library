import numpy as np
from scipy import sparse


def resultant(mtx):
    """
    :type param: sparse scipy matrix
    """
    if sparse.issparse(mtx):
        return np.array(mtx.sum(axis=0))[0]
    return mtx.sum(axis=0)


def resultant_cosine_sims(mtx):
    res = resultant(mtx)
    res_norm = np.linalg.norm(res)
    row_norms = sparse.linalg.norm(mtx, axis=1)
    return mtx.dot(res) / (row_norms * res_norm)


def col_count_from_col_lists(col_lists):
    """
    :type col_lists: list<list<int>>
    """
    maxcol = None
    for l in col_lists:
        if len(l) == 0:
            continue
        cur_max = max(l)
        if maxcol is None or maxcol < cur_max:
            maxcol = cur_max
    return maxcol


def mtx_from_col_lists(col_lists):
    """
    :type col_lists: list<list<int>>
    """
    numcols = col_count_from_col_lists(col_lists)
    mtx = sparse.lil_matrix((len(col_lists), numcols))
    for row in range(len(col_lists)):
        for col in col_lists[row]:
            mtx[row, col] += 1
    return mtx
