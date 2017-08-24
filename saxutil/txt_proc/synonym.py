import io
import csv
from lib.saxutil import pair_range_util


def default_match_genr_postag_method(from_tags, to_size):
    if from_tags[0] is None:
        return [None] * to_size
    starts = [t[0].lower() for t in from_tags]
    if 'v' in starts:
        # If verb is in "from_tags" assume "to_tags" are all verbs
        return ['v'] * to_size
    else:
        # Assume "to_tags" are all the final tag in "from_tags"
        return [from_tags[-1]] * to_size


class MatchGeneralizer(object):
    """
    A MatchGeneralizer instance remaps the values of a token or sequence of
    tokens to a different value. This is helpful when dealing with
    abbreviations and/or other languages. 

    A brief usage example can be found in lib/demo/mk_match_genr.py

    In that example I go over the code step by step.
    """
    def __init__(self, case_sensitive=False):
        self.from_ngrams_on_start = {}
        self.to_ngram_on_from_ngram = {}
        self.set_postag_method(default_match_genr_postag_method)
        self.case_sensitive = case_sensitive

    def set_postag_method(self, postag_method):
        self.postag_method = postag_method

    def add(self, from_ngram, to_ngram):
        """
        :type from_ngram: tuple<str>
        :type to_ngram: tuple<str>
        """
        if not self.case_sensitive:
            from_ngram = tuple([t.lower() for t in from_ngram])
            to_ngram = tuple([t.lower() for t in to_ngram])
        from_ngram, to_ngram = tuple(from_ngram), tuple(to_ngram)
        current_to_ngram = self.to_ngram_on_from_ngram.get(from_ngram)
        if current_to_ngram is not None:
            raise ValueError(
                'TO ngram is already present for the FROM ngram "' +
                ','.join([str(i) for i in to_ngram])
            )
        self.to_ngram_on_from_ngram[from_ngram] = to_ngram
        from_set = self.from_ngrams_on_start.get(from_ngram[0], set())
        from_set.add(from_ngram)
        self.from_ngrams_on_start[from_ngram[0]] = from_set

    def match_bounds(self, tkns):
        """
        :type tkns: list<str>
        """
        bnds = []
        for i in range(len(tkns)):
            candidate_ngrams = self.from_ngrams_on_start.get(tkns[i])
            if candidate_ngrams is None:
                continue

            match_ngrams = []
            for ng in candidate_ngrams:
                if i + len(ng) - 1 > len(tkns):
                    # ngram length exceeds the remaining length of the
                    # token sequence.
                    continue
                if tuple(tkns[i:i + len(ng)]) == ng:
                    match_ngrams.append(ng)

            for ng in match_ngrams:
                if tuple(tkns[i:i + len(ng)]) in candidate_ngrams:
                    bnds.append((i, i + len(ng) - 1,))

        subsumed = pair_range_util.subsumed_bound_indices(bnds)
        bounds_filt = []
        for i in range(len(bnds)):
            if i not in subsumed:
                bounds_filt.append(bnds[i])
        return bounds_filt

    def run(self, tkns, tags=None):
        """
        :type tkns: list<str>
        :type tags: list<str>
        """
        if tags is None:
            tags = [None] * len(tkns)
        if len(tags) != len(tkns):
            raise ValueError('len(tkns) != len(tags)')

        bnds = self.match_bounds(tkns)
        if len(bnds) == 0:
            return tkns, tags
        bnd_ix = 0
        proc_tkns = []
        proc_tags = []
        i = 0
        while i < len(tkns):
            if bnd_ix >= len(bnds) or bnds[bnd_ix][0] != i:
                proc_tkns.append(tkns[i])
                proc_tags.append(tags[i])
                i += 1
                continue
            beg, end = bnds[bnd_ix]
            from_ngram = tuple(tkns[beg:end + 1])
            to_ngram = self.to_ngram_on_from_ngram[from_ngram]
            from_tags = tuple(tags[beg:end + 1])
            to_tags = self.postag_method(from_tags, len(to_ngram))
            proc_tkns.extend(to_ngram)
            proc_tags.extend(to_tags)
            i += len(from_ngram)
            bnd_ix += 1
        return proc_tkns, proc_tags

    @classmethod
    def dumps(cls, match_generalizer):
        """
        :type match_generalizer: MatchGeneralizer
        :rtype: str
        """
        si = io.StringIO()
        wtr = csv.writer(si, delimiter='\t')
        items = list(match_generalizer.to_ngram_on_from_ngram.items())
        for from_ngram, to_ngram in items:
            row = list(from_ngram) + ['|'] + list(to_ngram)
            wtr.writerow(row)
        return si.getvalue().strip()

    @classmethod
    def loads(cls, dta, case_sensitive=False):
        """
        :type dta: str
        :rtype: MatchGeneralizer
        """
        si = io.StringIO(dta)
        rdr = csv.reader(si, delimiter='\t')
        mg = MatchGeneralizer(case_sensitive=case_sensitive)
        for row in rdr:
            split_ix = row.index('|')
            from_ngram = tuple(row[:split_ix])
            to_ngram = tuple(row[split_ix + 1:])
            mg.add(from_ngram, to_ngram)
        return mg

    @classmethod
    def load_from_file(cls, fpath):
        with open(fpath) as f:
            mg = MatchGeneralizer.loads(f.read())
        return mg
