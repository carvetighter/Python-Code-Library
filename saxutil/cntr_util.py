from scipy.sparse import lil_matrix
from scipy.spatial import distance


def convert_values_to_float(cntr):
    for k, v in cntr.items():
        cntr[k] = float(v)


def mtx_from_cntrs(cntrs):
    # Convert all the values in the counters to floats
    for i in range(len(cntrs)):
        convert_values_to_float(cntrs[i])

    # Map objects to matrix columns
    col_on_obj = {}
    cur_ix = 0
    for c in cntrs:
        for obj in c.keys():
            col = col_on_obj.get(obj)
            if col is None:
                col_on_obj[obj] = cur_ix
                cur_ix += 1
    numrows = len(cntrs)
    numcols = len(col_on_obj)
    mtx = lil_matrix((numrows, numcols))

    for rownum in range(numrows):
        for obj in cntrs[rownum].keys():
            colnum = col_on_obj[obj]
            mtx[rownum, colnum] = cntrs[rownum][obj]
    return mtx, col_on_obj


def dists_from_ref_cntr(ref_cntr, comp_cntrs):
    mtx, col_on_obj = mtx_from_cntrs([ref_cntr] + comp_cntrs)
    refvec = mtx.getrow(0).todense()
    dists = []
    for i in range(1, mtx.shape[0]):
        curvec = mtx.getrow(i).todense()
        dists.append(distance.cosine(refvec, curvec))
    return dists
