import nltk
from lib.saxutil import bowclf
from lib.saxutil.txt_proc import tkn_transform

class TrnrHighLvl(object):
    def __init__(self, pipe, trndocs, trnlbls):
        if len(trndocs) != len(trnlbls):
            raise ValueError('len(trndocs) != len(trnlbls)')
        self.trn_tkn_docs = [nltk.word_tokenize(d) for d in trndocs]
        self.trn_tkn_docs = (
            [tkn_transform.run_pipeline(pipe, self.trn_tkn_docs)]
        )
        self.trnlbls = trnlbls

    def mk_clf(self, unfitmdl, minfreq=1, scorepct=0):
        trnr = bowclf.Trnr()
        for i in range(len(self.trn_tkn_docs)):
            trnr.add_obj_list_doc(self.trn_tkn_docs[i], self.trnlbls[i])
        trnr.rm_fts(trnr.fts_below_freq(minfreq))
        trnr.rm_fts(trnr.lowest_score_pct_fts(scorepct))
        trnr.map_fts_to_cols()
        return trnr.to_clf(unfitmdl)