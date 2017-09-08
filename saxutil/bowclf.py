import os
import json
import pickle
import numpy as np
import nltk
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.feature_selection import SelectKBest, chi2
from collections import Counter
from lib.saxutil import ftmap
from lib.saxutil.txt_proc import tkn_transform


def map_fts_to_cols(fts):
    """
    :type fts: list<int>
    """
    fts = list(fts)
    col_on_ft = {}
    for i in range(len(fts)):
        col_on_ft[fts[i]] = i
    return col_on_ft


def mk_mtx(fmap, ft_cntrs, col_on_ft, row_norma=None):
    """
    :type fmap: FeatureMap
    :type ft_cntrs: list<Counter>
    :type col_on_ft: dict<int, int>
    :type row_norma: str
    :rtype: scipy.sparse.lil_matrix
    """
    numrows = len(ft_cntrs)
    numcols = len(col_on_ft)
    mtx = sparse.lil_matrix((numrows, numcols,))
    for row_ix in range(numrows):
        for ft, cnt in ft_cntrs[row_ix].items():
            col_ix = col_on_ft.get(ft)
            if col_ix is None:
                continue
            mtx[row_ix, col_ix] = cnt
    if row_norma is not None:
        return normalize(mtx, norm=row_norma)
    return mtx


def add_cntr_doc(cntr, fmap, cur_cntrs, cur_lbls=None, new_lbl=None):
    """
    :type cntr: Counter
    :type cur_cntrs: list<Counter>
    :type fmap: FeatureMap
    :type cur_lbls: list<obj>
    :type new_lbl: obj
    """
    ft_cntr = Counter()
    for obj in cntr:
        if cur_lbls is None:
            ft = fmap.get_ft(obj)
        else:
            ft = fmap.get_ft_add_on_absent(obj)
        if ft is None:
            continue
        ft_cntr[ft] += cntr[obj]
    cur_cntrs.append(ft_cntr)

    if cur_lbls is None:
        return
    cur_lbls.append(new_lbl)


def get_fts(ft_cntrs):
    fts = set()
    for fc in ft_cntrs:
        for f, c in fc.items():
            fts.add(f)
    return fts


class ClfBase(object):
    def __init__(self):
        self.fmap = ftmap.FeatureMap()
        self.ft_cntrs = []
        self.col_on_ft = None

    def clear(self):
        self.ft_cntrs = []

    def num_docs(self):
        return len(self.ft_cntrs)

    def mk_mtx(self, row_norma='l2'):
        return mk_mtx(
            self.fmap, self.ft_cntrs, self.col_on_ft, row_norma
        )

    def resultant_cntr(self):
        res = Counter()
        for i in range(len(self.ft_cntrs)):
            res += self.ft_cntrs[i]
        return res


class Clf(ClfBase):
    def __init__(self, mdl, fmap, col_on_ft, row_norma='l2'):
        """
        :param mdl: trained sklearn model
        :type fmap: FeatureMap
        :param fmap: feature map object that aligns words to matrix columns
        """
        super(Clf, self).__init__()
        self.mdl = mdl
        self.fmap = fmap
        self.ft_cntrs = []
        self.col_on_ft = col_on_ft
        self.row_norma = row_norma

    def num_fts(self):
        return len(self.col_on_ft)

    def add_cntr_doc(self, cntr):
        add_cntr_doc(cntr, self.fmap, self.ft_cntrs)

    def add_obj_list_doc(self, obj_list):
        self.add_cntr_doc(Counter(obj_list))

    def predict(self, return_proba=False):
        if not return_proba:
            return self.mdl.predict(self.mk_mtx(self.row_norma))

        pred_prob_mtx = [ [None, None] for i in range(self.num_docs()) ]
        #pred_prob_mtx = np.zeros((self.num_docs(), 2,))
        prob_arrs = self.mdl.predict_proba(self.mk_mtx(self.row_norma))
        for i in range(len(prob_arrs)):
            max_prob = max(prob_arrs[i])
            pred = self.mdl.classes_[list(prob_arrs[i]).index(max_prob)]
            pred_prob_mtx[i][0] = pred
            pred_prob_mtx[i][1] = max_prob
        return pred_prob_mtx


def serialize_clf_to_dir(clf, dirpath):
    if os.path.exists(dirpath):
        raise IOError('dirpath:\t' + str(dirpath) + ' already exists')
    os.mkdir(dirpath)
    with open(os.path.join(dirpath, 'fmap.json'), 'w') as f:
        f.write(ftmap.dumps(clf.fmap))
    with open(os.path.join(dirpath, 'col_on_ft.json'), 'w') as f:
        f.write(json.dumps(clf.col_on_ft))
    with open(os.path.join(dirpath, 'mdl.pkl'), 'wb') as f:
        f.write(pickle.dumps(clf.mdl))
    with open(os.path.join(dirpath, 'meta.json'), 'w') as f:
        f.write(json.dumps({'row_norma': clf.row_norma}))


def load_clf_from_dir(dirpath):
    with open(os.path.join(dirpath, 'fmap.json')) as f:
        fmap = ftmap.loads(f.read())
    with open(os.path.join(dirpath, 'col_on_ft.json')) as f:
        col_on_ft_loaded = json.loads(f.read())
        col_on_ft = {}
        for ft, col in col_on_ft_loaded.items():
            col_on_ft[int(ft)] = int(col)
    with open(os.path.join(dirpath, 'mdl.pkl'), 'rb') as f:
        mdl = pickle.loads(f.read())
    with open(os.path.join(dirpath, 'meta.json')) as f:
        meta = json.loads(f.read())
    return Clf(mdl, fmap, col_on_ft, meta['row_norma'])


class Trnr(ClfBase):
    def __init__(self):
        super(Trnr, self).__init__()
        self.lbls = []
        self.ft_bl = set()

    def num_fts(self):
        """
        col_on_ft may not be set. number of features in the FeatureMap (fmap)
        minus the length of the feature blacklist is equal to the number of
        features will be encoded in a generated matrix.
        """
        return self.fmap.num_fts() - len(self.ft_bl)

    def add_cntr_doc(self, cntr, lbl):
        """
        :type cntr: Counter
        :type lbl: obj
        """
        add_cntr_doc(cntr, self.fmap, self.ft_cntrs, self.lbls, lbl)

    def add_obj_list_doc(self, obj_list, lbl):
        """
        :type obj_list: list<obj>
        :param obj_list: in the basic unigram case, this will be a list of
            strings / words.
        :type lbl: obj
        """
        self.add_cntr_doc(Counter(obj_list), lbl)

    def add_fts_to_bl(self, fts):
        """
        :type fts: iterable<int>
        """
        for ft in fts:
            self.ft_bl.add(ft)

    def rm_fts(self, rmfts):
        """
        :type rmfts: set<int>
        """
        # column mappings can no longer be assumed to be valid
        self.col_on_ft = None
        # new feature map
        new_fmap = ftmap.rm_objs_remap(self.fmap, rmfts=rmfts)

        new_on_old = {}
        for old_ft in self.fmap.fts():
            o = self.fmap.get_obj(old_ft)
            new_ft = new_fmap.get_ft(o)
            if new_ft is not None:
                new_on_old[old_ft] = new_ft

        # new re-mapped feature blacklist
        new_ft_bl = set()
        for i in range(len(self.ft_cntrs)):
            new_ft_cntr = Counter()
            for old_ft in list(self.ft_cntrs[i].keys()):
                new_ft = new_on_old.get(old_ft)
                if new_ft is not None:
                    if old_ft in self.ft_bl:
                        new_ft_bl.add(new_ft)
                    new_ft_cntr[new_ft] = self.ft_cntrs[i][old_ft]
            self.ft_cntrs[i] = new_ft_cntr
        self.ft_bl = new_ft_bl
        self.fmap = new_fmap

    def score_fts(self, scorer=chi2, row_norma='l2'):
        self.map_fts_to_cols()
        ft_on_col = {f: c for c, f in self.col_on_ft.items()}
        mtx = self.mk_mtx(row_norma)
        self.col_on_ft = None
        sel = SelectKBest(scorer, k='all')
        sel.fit_transform(mtx, self.lbls)

        score_on_ft = {}
        for col in range(len(sel.scores_)):
            ft = ft_on_col[col]
            score_on_ft[ft] = sel.scores_[col]
        return score_on_ft

    def lowest_score_pct_fts(self, rm_proportion, scorer=chi2):
        """
        :type rm_proportion: float
        :param rm_proportion: number between 0 and 1
        :rtype: set<int>
        """
        if not 0.0 <= rm_proportion <= 1.0:
            raise ValueError('rm_proportion must be between 0 and 1')
        tups = list(self.score_fts(scorer).items())
        tups.sort(key=lambda t: t[1])
        lows = set()
        rm_end = int(len(tups) * rm_proportion)
        for i in range(0, rm_end):
            lows.add(tups[i][0])
        return lows

    def fts_below_freq(self, minfreq):
        """
        :type minfreq: int
        :rtype: set<int>
        """
        res = self.resultant_cntr()
        lows = set()
        for ft, cnt in res.items():
            if cnt < minfreq:
                lows.add(ft)
        return lows

    def map_fts_to_cols(self):
        ft_wl = self.fmap.fts() - self.ft_bl
        self.col_on_ft = map_fts_to_cols(ft_wl)

    def to_clf(self, mdl, row_norma='l2'):
        """
        :type mdl: sklearn.base.BaseEstimator
        :param mdl: untrained scikit-learn model
            EG: LogisticRegression(C=5)
        """
        mdl.fit(self.mk_mtx(row_norma), self.lbls)
        return Clf(mdl, self.fmap, self.col_on_ft, 'l2')


def add_trn_docs(docs, lbls, pipe, tokenize_method, trnr=None):
    if trnr is None:
        trnr = Trnr()
    if len(docs) != len(lbls):
        raise IndexError('len(docs) != len(lbls)')
    for i in range(len(docs)):
        tkns = tokenize_method(docs[i])
        ptl = tkn_transform.run_pipeline(pipe, tkns)[0]
        trnr.add_obj_list_doc(ptl, lbls[i])
    return trnr


def mk_bow_trnr(pipe, tkn_lil, tag_lil, lbls):
    if len(tkn_lil) != len(tag_lil) != len(lbls):
        raise ValueError('len(tkn_lil) != len(tag_lil) != len(lbls)')
    trnr = Trnr()
    for i in range(len(tkn_lil)):
        tkns, tags = tkn_transform.run_pipeline(pipe, tkn_lil[i], tag_lil[i])
        trnr.add_obj_list_doc(tkns, lbls[i])
    return trnr


def mk_bow_trnr_no_pos_tag(pipe, tkn_lil, lbls):
    tag_lil = [ [None] * len(d) for d in tkn_lil ]
    return mk_bow_trnr(pipe, tkn_lil, tag_lil, lbls)