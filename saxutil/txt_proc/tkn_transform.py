import re
#from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from lib.saxutil.txt_proc import regexes


def replace_run(replace_func, tkns, tags=None):
    """
    This is the base method the replacer classes use. The first parameter
    "replace_func" is a function that takes two paramters. The first ("tkn")
    is a token. The second ("tag") is a part of speech or POS tag.

    EXAMPLE
    -------
    def input_replace_func(tkn, tag):
        if tkn == 'mom': return 'dad', tag
        return tkn, tag

    tkns, tags = replace_func(input_replace_func, ['mom', 'is', 'great'])

    # The following two statements will be True
    tkns == ['dad', 'is', 'great']
    tags == [None, None, None]
    -------

    In the example above, all tokens equal to "mom" are replaced with "dad".
    The POS tags remain the same. As such, if the token "mom" and the POS tag
    None are passed into "input_replace_func", the method will return the
    token / POS tag pair: ("dad", None).

    When "input_replace_func" is passed into "replace_run" via the
    "replace_func" parameter, it is run over the token and tag lists denoted by
    parameters "tkns" and "tags" respectivey. "replace_run" applies the logic
    in "input_replace_func" to the list of tokens tags. The result is that the
    input token and tag lists: ['mom', 'is', 'great'] and [None, None, None]
    are transformed into ['dad', 'is', 'great'] and [None, None, None].

    NOTE: If the paramter "tags" is not specified, "replace_run" will set the
    "tags" parameter equal to a list of None or NULL values with the same
    length as the input tokens ("tkns"). 

    :type replace_func: function
    :param replace_func: use the logic in function to transform "tkns" and
        "tags".
    a token and tag as parameters.
    :type tkns: list<str>
    :type tags: list<str>
    """
    _tkns = [None] * len(tkns)
    _tags = [None] * len(tkns)
    if tags is None:
        # If no "tags" parameter is passed in, set to a list of None
        # values equal to the length of the "tkns" list.
        tags = [None] * len(tkns)
    assert(len(tkns) == len(tags))
    for i in range(len(tkns)):
        tkn, tag = replace_func(tkns[i], tags[i])
        _tkns[i], _tags[i] = tkn, tag
    assert(len(_tkns) == len(_tags) == len(tkns) == len(tags))
    return _tkns, _tags


def filt_run(should_reject_func, tkns, tags=None):
    """
    This is the base method the filter classes use. The first parameter
    "should_reject_func" is a function that takes two paramters. The first
    ("tkn") is a token. The second ("tag") is a part of speech or POS tag.

    EXAMPLE
    -------
    def input_should_reject(tkn, tag):
        if tkn in {'the', 'and', 'or'}:
            return True
        return False

    tkns, tags = filt_run(input_should_reject, ['the', 'dog', 'and', 'pony'])

    # The following statements will be true
    tkns == ['dog', 'pony']
    tags == [None, None]
    -------

    In the example above, any token equal to "the", "and", or "or" is rejected.
    The "filt_run" function generates a new list of tokens and POS tags that
    exclude all token / POS tag pairs where "input_shoud_reject" returns
    True.

    NOTE: If the paramter "tags" is not specified, "replace_run" will set the
    "tags" parameter equal to a list of None or NULL values with the same
    length as the input tokens ("tkns"). 

    :type should_reject_func: function
    :param should_reject_func: Function that returns True or False given a 
        token and tag as parameters. If True, reject. If False, keep. Use the
        logic in this method identify elements to remove from "tkns" and "tags".
    :type tkns: list<str>
    :type tags: list<str>
    """
    _tkns, _tags = [], []
    if tags is None:
        # If no "tags" parameter is passed in, set to a list of None
        # values equal to the length of the "tkns" list.
        tags = [None] * len(tkns)
    assert(len(tkns) == len(tags))
    for i in range(len(tkns)):
        reject = should_reject_func(tkns[i], tags[i])
        if not reject:
            _tkns.append(tkns[i])
            _tags.append(tags[i])
    assert(len(_tkns) == len(_tags))
    return _tkns, _tags


class FlagSplitter(object):
    """
    At TickerTags, the negation transform appends a flag on each
    negated token "__NEG". This created problems when trying to run additional
    transforms after the fact. The purpose of this class is to identify when
    a flag is present and account for prior to transformation.
    """

    def __init__(self, flgset):
        if len(flgset) == 0:
            self.patt = re.compile('$a')
            return
        pattstr = '('
        for flg in flgset:
            pattstr += flg + '|'
        pattstr = pattstr[:-1] + '$)'
        self.patt = re.compile(pattstr)

    def split(self, tkn):
        res = self.patt.search(tkn)
        if res is None:
            return tkn, ''
        return tkn[:res.start()], tkn[res.start():len(tkn)]


class TransformBase(object):
    def __init__(self, flgset, run_method):
        self.flg_splitr = FlagSplitter(flgset)
        self.run_method = run_method

    def run(self, tkns, tags=None):
        return self.run_method(self.tkn_processor, tkns, tags)


class LowerCaseReplacer(TransformBase):
    """
    Usage: Replace all tokens with their lower case equivalent.
    """
    def __init__(self, flgset=set()):
        super(LowerCaseReplacer, self).__init__(flgset, replace_run)

    def tkn_processor(self, tkn, tag=None):
        rtkn, flg = self.flg_splitr.split(tkn)
        return rtkn.lower() + flg, tag


class StemReplacer(TransformBase):
    """
    Usage: Replace all tokens with thier stemmed form.
    """
    def __init__(self, stemmer=PorterStemmer(), flgset=set()):
        super(StemReplacer, self).__init__(flgset, replace_run)
        self.stemmer = stemmer

    def tkn_processor(self, tkn, tag=None):
        rtkn, flg = self.flg_splitr.split(tkn)
        return self.stemmer.stem(rtkn) + flg, tag


class LemmaReplacer(TransformBase):
    def __init__(self, flgset=set()):
        super(LemmaReplacer, self).__init__(flgset, replace_run)
        self.wnl = WordNetLemmatizer()
        self.wn_pos_on_tbnk_pos_start = {
            'A':'a', 'R':'r', 'V':'v'
        }

    def tkn_processor(self, tkn, tag):
        if tag is None or len(tag) == 0:
            raise ValueError('POS tag required')
        wnpos = self.wn_pos_on_tbnk_pos_start.get(tag[0], 'n')
        rtkn = self.wnl.lemmatize(tkn, wnpos)
        return rtkn, tag


class NumberReplacer(TransformBase):
    """
    Usage: Set all numeric tokens to a uniform string.
    """
    def __init__(self, flgset=set(), replace_with='__numeric'):
        super(NumberReplacer, self).__init__(flgset, replace_run)
        self.replace_with = replace_with

    def tkn_processor(self, tkn, tag=None):
        rtkn, flg = self.flg_splitr.split(tkn)
        if regexes.IS_NUMERIC_PATT.match(rtkn):
            return self.replace_with + flg, tag
        return rtkn + flg, tag


class TermWithPosReplacer(TransformBase):
    def __init__(self, flgset=set()):
        super(TermWithPosReplacer, self).__init__(flgset, replace_run)
        self.to_on_from = {}

    def add_tkn_pos_transform(self, tkn, tag, to_tkn):
        self.to_on_from[(tkn, tag,)] = to_tkn

    def tkn_processor(self, tkn, tag):
        sptkn, flg = self.flg_splitr.split(tkn)
        rtkn = self.to_on_from.get( (sptkn, tag,), sptkn )
        return rtkn, tag


class StopWordFilter(TransformBase):
    """
    Usage: Remove stopwords
    """
    def __init__(self, stopwords, flgset=set()):
        super(StopWordFilter, self).__init__(flgset, filt_run)
        self.stopwords = set(stopwords)

    def tkn_processor(self, tkn, tag=None):
        rtkn, flg = self.flg_splitr.split(tkn)
        if rtkn in self.stopwords:
            return True
        return False


class NumberFilter(TransformBase):
    """
    Usage: Remove numeric tokens.
    """
    def __init__(self, flgset=set()):
        super(NumberFilter, self).__init__(flgset, filt_run)

    def tkn_processor(self, tkn, tag=None):
        rtkn, flg = self.flg_splitr.split(tkn)
        if regexes.IS_NUMERIC_PATT.match(rtkn):
            return True
        return False


class MatchGeneralizeTransformer(object):
    """
    This transformer does not support flag identification.
    As such, it must be run prior to any transform that flags tokens
    """
    def __init__(self, match_generalizer):
        self.mg = match_generalizer

    def run(self, tkns, tags=None):
        ptkns, ptags = self.mg.run(tkns, tags)
        return ptkns, ptags


def run_pipeline(transformers, tkns, tags=None):
    if tags is None:
        tags = [None] * len(tkns)

    _tkns, _tags = list(tkns), list(tags)
    for tr in transformers:
        _tkns, _tags = tr.run(_tkns, _tags)
    return _tkns, _tags
